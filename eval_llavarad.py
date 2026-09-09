#!/usr/bin/env python3
# eval_llavarad.py -- LLaVA-Rad IU chest-X-ray evaluation
#
# Modes:
#   baseline  Re-score baseline_preds.jsonl on CPU. MUST reproduce the
#             original RadGraph ~0.287; that validates the metric path.
#   generate  Load the fine-tuned model (4-bit + LoRA, NO merge) and write
#             finetuned_preds.jsonl. Append-safe: rerun skips done ids.
#   score     Score finetuned_preds.jsonl on CPU, write metrics.json and
#             review_shortlist.jsonl (~16 samples for manual review).
#   all       baseline + generate + score (the normal run).
#
# Examples:
#   CUDA_VISIBLE_DEVICES=1 ~/llavarad-env/bin/python ~/iu-xray/eval_llavarad.py --mode baseline
#   CUDA_VISIBLE_DEVICES=1 ~/llavarad-env/bin/python ~/iu-xray/eval_llavarad.py --mode generate \
#       --limit 2 --adapter-dir ~/llava-rad-checkpoints/llavarad-iu/checkpoint-300
#   CUDA_VISIBLE_DEVICES=1 ~/llavarad-env/bin/python ~/iu-xray/eval_llavarad.py --mode all
#
# Generation settings are copied verbatim from the official eval
# (llava/eval/model_mimic_cxr.py): greedy (temperature 0), max_new_tokens 256,
# num_beams 1, conversation template v1, prompt = IMAGE_TOKEN + "\n" + query,
# images cast to float16 on cuda.
#
# Model load mirrors the "llavarad" LoRA branch of llava/model/builder.py
# (4-bit nf4 double-quant, base vicuna-7b-v1.5, config from the adapter dir,
# non_lora_trainables.bin = fine-tuned mm projector, PeftModel) but SKIPS the
# builder's merge_and_unload(): merging would upcast every LoRA layer to
# 16-bit and blow the 16 GB VRAM budget. Expected VRAM: ~4 GB LLM (nf4) +
# ~0.3 GB LoRA + ~1.5 GB BiomedCLIP-CXR (fp16) + activations < 9 GB total.

import argparse
import json
import os
import random
import sys
import time

HOME = os.path.expanduser("~")
IU = os.path.join(HOME, "iu-xray")
REPO = os.path.join(HOME, "LLaVA-Rad")
CKPT = os.path.join(HOME, "llava-rad-checkpoints")

BASE_FILE = os.path.join(IU, "baseline_preds.jsonl")
FT_FILE = os.path.join(IU, "finetuned_preds.jsonl")
TEST_JSON = os.path.join(IU, "llava_iu", "test.json")
IMAGE_DIR = os.path.join(IU, "llava_iu", "images")
METRICS_JSON = os.path.join(IU, "metrics.json")
SHORTLIST = os.path.join(IU, "review_shortlist.jsonl")
MODEL_BASE = os.path.join(CKPT, "vicuna-7b-v1.5")
ADAPTER_DIR = os.path.join(CKPT, "llavarad-iu")
VIT_CONFIG = os.path.join(CKPT, "llavarad", "biomedclipcxr_518.json")
VIT_CKPT = os.path.join(CKPT, "llavarad", "biomedclipcxr_518_checkpoint.pt")
MM_PROJECTOR_BIN = os.path.join(CKPT, "llavarad", "mm_projector.bin")
RG_KEYS = ["rg_e", "rg_er", "rg_bar_er"]
ORIG_BASELINE_RG = 0.287  # number to beat (and to reproduce)


def load_jsonl(path):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def load_data():
    """Return (id->test record, id->baseline record). Checks ids + images."""
    with open(TEST_JSON) as f:
        test = json.load(f)
    test_map = {r["id"]: r for r in test}
    base_map = {r["id"]: r for r in load_jsonl(BASE_FILE)}
    missing = [i for i in base_map if i not in test_map]
    if missing:
        raise SystemExit("ERROR: %d baseline ids missing from test.json, e.g. %s"
                         % (len(missing), missing[0]))
    mismatch = [i for i in base_map
                if base_map[i]["query"].strip()
                != test_map[i]["conversations"][0]["value"].strip()]
    if mismatch:
        print("WARNING: query differs between baseline file and test.json for %d ids"
              % len(mismatch))
    missing_img = [test_map[i]["image"] for i in base_map
                   if not os.path.exists(os.path.join(IMAGE_DIR, test_map[i]["image"]))]
    if missing_img:
        raise SystemExit("ERROR: %d image files missing, e.g. %s"
                         % (len(missing_img), missing_img[0]))
    print("data ok: %d baseline records, %d test records, all images present"
          % (len(base_map), len(test_map)))
    return test_map, base_map

# ----------------------------- metrics ----------------------------------


def _lcs_len(a, b):
    """Length of the longest common subsequence of token lists a, b."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        cur = [0] * (n + 1)
        ai = a[i - 1]
        for j in range(1, n + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = cur[j - 1] if cur[j - 1] >= prev[j] else prev[j]
        prev = cur
    return prev[n]


def rouge_l_f1(hyp, ref):
    """Word-level ROUGE-L F1 via LCS (no external package)."""
    h = hyp.strip().split()
    r = ref.strip().split()
    if not h or not r:
        return 0.0
    l = _lcs_len(h, r)
    if l == 0:
        return 0.0
    p, rec = l / len(h), l / len(r)
    return 2.0 * p * rec / (p + rec)


def rg_score(hyps, refs, args):
    """Batch RadGraph score via pypi `radgraph` 0.1.18 F1RadGraph.

    Verified return shape (radgraph/radgraph.py F1RadGraph.forward):
      reward_level="all":                     3 floats      + (list, list, list)  # 3 x N
      reward_level="simple"/"partial"/"complete": one float + list of N floats
    Normalized to [rg_e, rg_er, rg_bar_er] = (simple, partial, complete).
    Returns {"mean": [3 floats], "per": [[3 floats] x N], "how": str}.
    """
    from radgraph import F1RadGraph
    metric = F1RadGraph(reward_level=args.rg_reward, model_type=args.rg_model)
    how = "F1RadGraph(reward_level='%s', model_type='%s')" % (
        args.rg_reward, args.rg_model)
    n = len(hyps)
    mean, raw_per = metric(hyps=list(hyps), refs=list(refs))[0:2]
    if args.rg_reward == "all":
        if len(raw_per) != 3 or any(len(c) != n for c in raw_per):
            raise SystemExit("ERROR: radgraph 'all' per-sample shape %s, expected 3 x %d"
                             % ([len(c) for c in raw_per], n))
        per = [[float(raw_per[c][i]) for c in range(3)] for i in range(n)]
        m = [float(mean[c]) for c in range(3)]
    else:
        slot = {"simple": 0, "partial": 1, "complete": 2}[args.rg_reward]
        if len(raw_per) != n:
            raise SystemExit("ERROR: radgraph returned %d per-sample scores for %d hyps"
                             % (len(raw_per), n))
        per = [[0.0] * 3 for _ in range(n)]
        for i, v in enumerate(raw_per):
            per[i][slot] = float(v)
        m = [0.0, 0.0, 0.0]
        m[slot] = float(mean)
    return {"mean": m, "per": per, "how": how}


def compute_scores(recs, label, args):
    """ROUGE-L + RadGraph for a list of {id,reference,prediction} records."""
    preds = [r["prediction"] for r in recs]
    refs = [r["reference"] for r in recs]
    rl = [rouge_l_f1(p, r) for p, r in zip(preds, refs)]
    print("[%s] n=%d  ROUGE-L F1 = %.4f" % (label, len(recs), sum(rl) / len(rl)),
          flush=True)
    t0 = time.time()
    rg = rg_score(preds, refs, args)
    m = rg["mean"]
    print("[%s] RadGraph %s  (%.0fs)" % (label, rg["how"], time.time() - t0),
          flush=True)
    print("[%s]   rg_e=%.4f  rg_er=%.4f  rg_bar_er=%.4f" % ((label,) + tuple(m)),
          flush=True)
    return {
        "rouge_l": sum(rl) / len(rl),
        "rouge_l_per": {r["id"]: x for r, x in zip(recs, rl)},
        "rg": {k: m[i] for i, k in enumerate(RG_KEYS)},
        "rg_per": {r["id"]: rg["per"][i] for i, r in enumerate(recs)},
        "rg_how": rg["how"],
        "n": len(recs),
    }


def pick_headline(base_block):
    """The RadGraph component closest to the original 0.287 = headline."""
    means = base_block["rg"]
    return min(RG_KEYS, key=lambda k: abs(means[k] - ORIG_BASELINE_RG))

# --------------------------- model + generation -------------------------


def _install_projector(model, dense):
    """Install the mm_projector as a fresh fp16 mlp2x_gelu with these weights.

    bnb 4-bit quantizes every Linear at load time (projector included), and
    dense weights cannot be copied into a quantized module. The projector was
    trained in fp16, so fp16 is the correct (training-time) precision.
    """
    import torch
    import torch.nn as nn
    proj = nn.Sequential(
        nn.Linear(dense["0.weight"].shape[1], model.config.hidden_size),
        nn.GELU(),
        nn.Linear(model.config.hidden_size, model.config.hidden_size),
    )
    missing, unexpected = proj.load_state_dict(dense)
    if missing or unexpected:
        raise SystemExit("projector mismatch: missing=%s unexpected=%s"
                         % (missing, unexpected))
    model.model.mm_projector = proj.to(device="cuda", dtype=torch.float16)
    print("  projector installed (fp16): in=%d out=%d"
          % (dense["0.weight"].shape[1], model.config.hidden_size), flush=True)


def load_model(adapter_dir, device):
    import torch
    from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    from llava.model import LlavaLlamaForCausalLM  # import also registers "llava"

    cfg_file = os.path.join(adapter_dir, "config.json")
    if os.path.exists(cfg_file):
        cfg = AutoConfig.from_pretrained(adapter_dir)
        # The final save embedded the 4-bit quantization_config in this file.
        # transformers' reconstruction of it is broken (private _load_in_4bit
        # field dropped -> load_in_4bit=False) and it would hijack the
        # explicit BitsAndBytesConfig below ("config wins") and switch the
        # loader into pre-quantized-checkpoint mode -> strip it.
        if getattr(cfg, "quantization_config", None) is not None:
            cfg.quantization_config = None
            print("note: stripped embedded quantization_config from saved "
                  "config (re-quantizing explicitly below)", flush=True)
        print("config: " + cfg_file, flush=True)
    else:
        cfg = AutoConfig.from_pretrained(MODEL_BASE)
        cfg.model_type = "llava"
        cfg.use_mm_proj = True
        cfg.mm_projector_type = "mlp2x_gelu"
        cfg.mm_hidden_size = 768
        cfg.mm_vision_select_layer = -2
        cfg.mm_vision_select_feature = "patch"
        cfg.mm_use_im_start_end = False
        cfg.mm_use_im_patch_token = False
        print("WARNING: no config.json in adapter dir -> built llava config from "
              "the base (fine for a checkpoint-N smoke test; the final dir "
              "has config.json)", flush=True)
    # Vision routing: same values training used (belt and suspenders).
    cfg.mm_vision_tower = "biomedclip_cxr_518"
    cfg.mm_vision_tower_config = VIT_CONFIG
    cfg.mm_vision_tower_checkpoint = VIT_CKPT

    tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE, use_fast=False)
    print("loading base LLM (4-bit nf4 double-quant) ...", flush=True)
    model = LlavaLlamaForCausalLM.from_pretrained(
        MODEL_BASE, low_cpu_mem_usage=True, config=cfg, device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4"))
    model.resize_token_embeddings(len(tokenizer))

    non_lora_file = os.path.join(adapter_dir, "non_lora_trainables.bin")
    if os.path.exists(non_lora_file):
        print("loading fine-tuned projector: " + non_lora_file, flush=True)
        proj_sd = torch.load(non_lora_file, map_location="cpu", weights_only=False)
        # same key normalization as the official builder (model/builder.py)
        proj_sd = {(k[11:] if k.startswith("base_model.") else k): v
                   for k, v in proj_sd.items()}
        if any(k.startswith("model.model.") for k in proj_sd):
            proj_sd = {(k[6:] if k.startswith("model.") else k): v
                       for k, v in proj_sd.items()}
    else:
        print("WARNING: no non_lora_trainables.bin in %s -> using the official "
              "PRE-TRAINING projector (smoke test only; the final dir has "
              "the fine-tuned projector)" % adapter_dir, flush=True)
        proj_sd = torch.load(MM_PROJECTOR_BIN, map_location="cpu",
                             weights_only=False)
    dense = {}
    for k, v in proj_sd.items():
        if "mm_projector" in k:
            dense[".".join(k.split(".")[-2:])] = v  # e.g. 0.weight / 2.bias
    if set(dense) != {"0.weight", "0.bias", "2.weight", "2.bias"}:
        raise SystemExit("ERROR: unexpected projector keys: %s" % sorted(dense))
    _install_projector(model, dense)

    print("loading LoRA adapter: " + adapter_dir, flush=True)
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    vt = model.get_vision_tower()
    if not vt.is_loaded:
        print("loading BiomedCLIP-CXR vision tower (frozen, fp16) ...", flush=True)
        vt.load_model()
    vt.to(device=device, dtype=torch.float16)
    print("model ready. VRAM used by this process: %.2f GB (budget 16 GB)"
          % (torch.cuda.max_memory_allocated() / 1e9), flush=True)
    return model, tokenizer, vt.image_processor


def run_generate(args, test_map, base_map):
    import torch
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    from llava.conversation import conv_templates, SeparatorStyle
    from llava.mm_utils import tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.utils import disable_torch_init
    disable_torch_init()

    device = "cuda"
    model, tokenizer, image_processor = load_model(args.adapter_dir, device)
    cv0 = conv_templates["v1"]
    stop_str = cv0.sep if cv0.sep_style != SeparatorStyle.TWO else cv0.sep2

    done = set()
    if os.path.exists(FT_FILE):
        for r in load_jsonl(FT_FILE):
            done.add(r["id"])
    todo = [rid for rid in base_map if rid not in done]
    if args.limit > 0:
        todo = todo[:args.limit]
    print("generate: %d to do (%d already in %s)" % (len(todo), len(done), FT_FILE),
          flush=True)
    if not todo:
        return

    t0 = time.time()
    with open(FT_FILE, "a") as out:
        for i, rid in enumerate(todo, 1):
            base_rec = base_map[rid]
            q = base_rec["query"].replace("<image>", "").strip()
            q = DEFAULT_IMAGE_TOKEN + "\n" + q
            conv = cv0.copy()
            conv.append_message(conv.roles[0], q)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                              return_tensors="pt")  # 1D (seq_len,)
            image = Image.open(os.path.join(IMAGE_DIR, test_map[rid]["image"])).convert("RGB")
            image_tensor = image_processor.preprocess(image, return_tensors="pt")[
                "pixel_values"][0]
            # single sample still needs a batch dim: unsqueeze(0) == the official
            # eval's torch.stack() over a batch of one (2D input_ids, 3D images)
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids.unsqueeze(0).to(device),
                    images=image_tensor.half().unsqueeze(0).to(device),
                    do_sample=False, temperature=0, num_beams=1,
                    max_new_tokens=args.max_new_tokens, use_cache=True).cpu()
            out_ids = output_ids[0][input_ids.shape[0]:].tolist()
            if tokenizer.eos_token_id is not None and tokenizer.eos_token_id in out_ids:
                out_ids = out_ids[:out_ids.index(tokenizer.eos_token_id)]
            pred = tokenizer.decode(out_ids, skip_special_tokens=True).strip()
            if pred.endswith(stop_str):
                pred = pred[:-len(stop_str)].strip()
            out.write(json.dumps({"id": rid, "query": base_rec["query"],
                                  "reference": base_rec["reference"],
                                  "prediction": pred}) + "\n")
            out.flush()
            el = time.time() - t0
            print("[%d/%d] %s pred=%dw avg=%.1fs eta=%.1fmin"
                  % (i, len(todo), rid, len(pred.split()), el / i,
                     el / i * (len(todo) - i) / 60), flush=True)
    print("generate done in %.1f min -> %s" % ((time.time() - t0) / 60, FT_FILE),
          flush=True)

# --------------------------- scoring + shortlist ------------------------


def run_score(args, base_block, test_map, base_map):
    recs = load_jsonl(FT_FILE)
    if len(recs) != len(base_map):
        print("WARNING: finetuned file has %d records, baseline has %d"
              % (len(recs), len(base_map)))
    ft_block = compute_scores(recs, "finetuned", args)
    headline = pick_headline(base_block)
    hi = RG_KEYS.index(headline)

    print("")
    print("== RESULTS (headline metric: %s = closest to the original %.3f) =="
          % (headline, ORIG_BASELINE_RG))
    for name, block in (("baseline", base_block), ("finetuned", ft_block)):
        m = block["rg"]
        print("%-9s rg_e=%.4f rg_er=%.4f rg_bar_er=%.4f rouge_l=%.4f"
              % (name, m["rg_e"], m["rg_er"], m["rg_bar_er"], block["rouge_l"]))
    delta = ft_block["rg"][headline] - base_block["rg"][headline]
    print("delta %s (finetuned - baseline): %+.4f" % (headline, delta))

    rows = []
    for rid in [r["id"] for r in recs]:
        b = base_block["rg_per"][rid][hi]
        f = ft_block["rg_per"][rid][hi]
        rows.append({"id": rid, "b": b, "f": f, "d": f - b,
                     "rl_b": base_block["rouge_l_per"][rid],
                     "rl_f": ft_block["rouge_l_per"][rid]})
    by_id = {r["id"]: r for r in recs}
    picked = {}
    for r in sorted(rows, key=lambda x: -x["d"])[:5]:
        picked.setdefault(r["id"], "improved")
    for r in sorted(rows, key=lambda x: x["d"])[:5]:
        picked.setdefault(r["id"], "dropped")
    for r in sorted(rows, key=lambda x: (x["b"] + x["f"]) / 2)[:4]:
        picked.setdefault(r["id"], "low_both")
    for rid in random.Random(42).sample(list(by_id), 2):
        picked.setdefault(rid, "random")

    def order_key(item):
        rid, why = item
        r = next(x for x in rows if x["id"] == rid)
        if why == "improved":
            return (0, -r["d"])
        if why == "dropped":
            return (1, r["d"])
        return (2, r["b"] + r["f"])

    ordered = sorted(picked.items(), key=order_key)
    with open(SHORTLIST, "w") as f:
        for rid, why in ordered:
            r = next(x for x in rows if x["id"] == rid)
            b = base_map[rid]
            rec = {"id": rid,
                   "image": test_map[rid]["image"] if rid in test_map else "",
                   "query": b["query"],
                   "reference": b["reference"],
                   "baseline_prediction": b["prediction"],
                   "finetuned_prediction": by_id[rid]["prediction"],
                   "baseline_" + headline: r["b"],
                   "finetuned_" + headline: r["f"],
                   "delta_" + headline: r["d"],
                   "baseline_rouge_l_f1": r["rl_b"],
                   "finetuned_rouge_l_f1": r["rl_f"],
                   "reason": why}
            f.write(json.dumps(rec) + "\n")
    print("shortlist: %d samples -> %s" % (len(ordered), SHORTLIST))

    metrics = {
        "baseline": {k: v for k, v in base_block.items() if not k.endswith("_per")},
        "finetuned": {k: v for k, v in ft_block.items() if not k.endswith("_per")},
        "headline_metric": headline,
        "delta_headline": delta,
        "notes": ("generation: greedy, max_new_tokens=%d, conv v1; model: 4-bit "
                  "nf4 double-quant + LoRA r=64 unmerged, BiomedCLIP-CXR frozen; "
                  "rouge_l: whitespace-token LCS, case-sensitive"
                  % args.max_new_tokens),
    }
    with open(METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2)
    print("metrics -> " + METRICS_JSON)


def main():
    ap = argparse.ArgumentParser(description="LLaVA-Rad IU evaluation")
    ap.add_argument("--mode", choices=["baseline", "generate", "score", "all"],
                    default="all")
    ap.add_argument("--adapter-dir", default=ADAPTER_DIR,
                    help="dir with adapter_model.safetensors (final run: "
                         "llava-rad-checkpoints/llavarad-iu; smoke test: its "
                         "checkpoint-300 subdir)")
    ap.add_argument("--limit", type=int, default=0,
                    help="generate only: max samples to generate this run (0=all)")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--rg-reward", choices=["all", "partial"], default="all",
                    help="radgraph reward_level (relation-reward scheme)")
    ap.add_argument("--rg-model", default="radgraph-xl",
                    help="radgraph F1RadGraph model_type")
    args = ap.parse_args()
    args.adapter_dir = os.path.expanduser(args.adapter_dir)
    if REPO not in sys.path:
        sys.path.insert(0, REPO)

    test_map, base_map = load_data()

    base_block = None
    if args.mode in ("baseline", "all", "score"):
        base_block = compute_scores(load_jsonl(BASE_FILE), "baseline", args)

    if args.mode in ("generate", "all"):
        if not os.path.exists(os.path.join(args.adapter_dir,
                                           "adapter_model.safetensors")):
            raise SystemExit("ERROR: no adapter_model.safetensors in %s\n"
                             "  has training finished? for a smoke test use:\n"
                             "  --adapter-dir %s/checkpoint-300"
                             % (args.adapter_dir, args.adapter_dir))
        run_generate(args, test_map, base_map)

    if args.mode in ("score", "all"):
        run_score(args, base_block, test_map, base_map)

    if args.mode == "baseline":
        m = {k: v for k, v in base_block.items() if not k.endswith("_per")}
        with open(METRICS_JSON, "w") as f:
            json.dump({"baseline": m}, f, indent=2)
        close = pick_headline(base_block)
        print("")
        print("metrics (baseline only) -> " + METRICS_JSON)
        print("closest to the original %.3f: %s = %.4f"
              % (ORIG_BASELINE_RG, close, base_block["rg"][close]))
        print("if NONE of the three is close to 0.287, the original run used a")
        print("different radgraph config -- paste me the three numbers above.")


if __name__ == "__main__":
    main()
