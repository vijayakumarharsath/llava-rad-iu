#!/usr/bin/env python3
"""
run_v3_eval.py -- v3 (bf16, no quantization) generate + score wrapper.

Why this exists: frozen eval_llavarad.py (v7) is right for v1/v2, but its
load_model() hardcodes a 4-bit nf4 double-quant base. The v3 adapter was
trained against a BF16 base, so evaluating it against a 4-bit base would
compare against a numerically different model (and would contradict the
no-quantization premise of v3). Everything else -- prompt building, conv
template, greedy 256-token generation, append-safe resume, ROUGE-L,
RadGraph, metric JSON shape -- is reused from the frozen file, unmodified.

What this wrapper does (in-process monkeypatches, no file edits):
  1. verifies eval_llavarad.py sha256 against the frozen v7 hash
  2. ev.FT_FILE       -> finetuned_preds_{tag}.jsonl
     ev.METRICS_JSON  -> metrics_{tag}.json
     ev.SHORTLIST     -> review_shortlist_{tag}.jsonl
     ev.load_model    -> bf16 variant (no BitsAndBytes; projector installed
                         in bf16 to match v3 training dtypes; vision bf16
                         and pixels cast to bf16 at generate, matching the
                         exact dtype mix v3 trained with; device_map=auto
                         over the visible GPUs)
  3. frozen run_generate  (greedy, max_new_tokens=256, append-safe)
  4. frozen run_score     (baseline + finetuned, headline pick, metrics json)
     + notes string rewritten to describe the bf16 model
  5. frozen eval_extra.run_findings + run_entities, unclean AND raycleaned,
     outputs versioned: findings_metrics_{tag}[clean].json,
     entity_metrics_{tag}[clean].json. eval_extra imports the SAME module
     object, so it sees the patched FT_FILE/METRICS_JSON automatically.

Usage (one invocation per checkpoint; both GPUs visible, training must be
finished so the GPUs are free):
  ~/llavarad-env/bin/python run_v3_eval.py --tag v3_531 \
      --adapter-dir ~/llava-rad-checkpoints/llava-rad-iu-v3/checkpoint-531
  ~/llavarad-env/bin/python run_v3_eval.py --tag v3_final \
      --adapter-dir ~/llava-rad-checkpoints/llava-rad-iu-v3
Quick 3-sample generate smoke:  ... --tag v3_smoke --limit 3
Re-score an existing preds file: ... --skip-generate
"""

import argparse
import hashlib
import json
import os
import sys

FROZEN_SHA = "0b88eaaea85daef74901fa7e16873942aaaaffbb686167bb544b07aa389a9ad9"


def check_frozen(here):
    path = os.path.join(here, "eval_llavarad.py")
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if h != FROZEN_SHA:
        raise SystemExit("ERROR: eval_llavarad.py sha %s != frozen %s "
                         "(v7 must stay untouched)" % (h[:12], FROZEN_SHA[:12]))
    print("eval_llavarad.py frozen-sha OK (%s...)" % h[:12], flush=True)


def make_bf16_load_model():
    """The frozen load_model with exactly three changes: (1) no 4-bit
    quantization (bf16 base instead); (2) projector installed in bf16 (v3
    trained it in bf16; the frozen path used fp16 because v1/v2 trained on
    a 4-bit base with fp16 compute); (3) the exact dtype mix v3 trained
    with - vision tower bf16 (train.py line 921: vision_tower.to(bfloat16)
    under --bf16) and bf16 pixels at the model boundary (dataset/collator
    emit fp32 pixel_values; no explicit cast in train.py; the bf16
    projector survived all 8496 training forwards, so features reached it
    in bf16). The frozen run_generate passes fp16 pixels, so
    load_model_bf16 also wraps model.generate to cast the images kwarg to
    bf16. Everything else is copied verbatim."""
    def _install_projector(model, dense, dtype):
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
        model.model.mm_projector = proj.to(device="cuda", dtype=dtype)
        print("  projector installed (%s): in=%d out=%d"
              % (dtype, dense["0.weight"].shape[1], model.config.hidden_size),
              flush=True)

    def load_model_bf16(adapter_dir, device):
        import torch
        from transformers import AutoConfig, AutoTokenizer
        from peft import PeftModel
        from llava.model import LlavaLlamaForCausalLM  # import also registers "llava"

        cfg_file = os.path.join(adapter_dir, "config.json")
        if os.path.exists(cfg_file):
            cfg = AutoConfig.from_pretrained(adapter_dir)
            if getattr(cfg, "quantization_config", None) is not None:
                cfg.quantization_config = None
                print("note: stripped embedded quantization_config from saved "
                      "config (should not be present in a v3 dir)", flush=True)
            print("config: " + cfg_file, flush=True)
        else:
            cfg = AutoConfig.from_pretrained(ev.MODEL_BASE)
            cfg.model_type = "llava"
            cfg.use_mm_proj = True
            cfg.mm_projector_type = "mlp2x_gelu"
            cfg.mm_hidden_size = 768
            cfg.mm_vision_select_layer = -2
            cfg.mm_vision_select_feature = "patch"
            cfg.mm_use_im_start_end = False
            cfg.mm_use_im_patch_token = False
            print("note: no config.json in adapter dir -> built llava config "
                  "from the base (normal for a checkpoint-N dir; the final "
                  "dir saves config.json)", flush=True)
        # Vision routing: same values training used (belt and suspenders).
        cfg.mm_vision_tower = "biomedclip_cxr_518"
        cfg.mm_vision_tower_config = ev.VIT_CONFIG
        cfg.mm_vision_tower_checkpoint = ev.VIT_CKPT

        tokenizer = AutoTokenizer.from_pretrained(ev.MODEL_BASE, use_fast=False)
        print("loading base LLM (bf16, NO quantization, device_map=auto over "
              "%d visible GPU(s)) ..." % torch.cuda.device_count(), flush=True)
        model = LlavaLlamaForCausalLM.from_pretrained(
            ev.MODEL_BASE, low_cpu_mem_usage=True, config=cfg,
            device_map="auto", torch_dtype=torch.bfloat16)
        model.resize_token_embeddings(len(tokenizer))

        non_lora_file = os.path.join(adapter_dir, "non_lora_trainables.bin")
        if os.path.exists(non_lora_file):
            print("loading fine-tuned projector: " + non_lora_file, flush=True)
            proj_sd = torch.load(non_lora_file, map_location="cpu",
                                 weights_only=False)
            # same key normalization as the official builder (model/builder.py)
            proj_sd = {(k[11:] if k.startswith("base_model.") else k): v
                       for k, v in proj_sd.items()}
            if any(k.startswith("model.model.") for k in proj_sd):
                proj_sd = {(k[6:] if k.startswith("model.") else k): v
                           for k, v in proj_sd.items()}
        else:
            raise SystemExit(
                "ERROR: no non_lora_trainables.bin in %s. v3 checkpoints save "
                "it in the checkpoint dir; refusing to fall back to the "
                "pre-training projector (that would not be the v3 model)."
                % adapter_dir)
        dense = {}
        for k, v in proj_sd.items():
            if "mm_projector" in k:
                dense[".".join(k.split(".")[-2:])] = v  # e.g. 0.weight / 2.bias
        if set(dense) != {"0.weight", "0.bias", "2.weight", "2.bias"}:
            raise SystemExit("ERROR: unexpected projector keys: %s" % sorted(dense))
        _install_projector(model, dense, dtype=torch.bfloat16)

        print("loading LoRA adapter: " + adapter_dir, flush=True)
        model = PeftModel.from_pretrained(model, adapter_dir)
        model.eval()

        # V3 EVAL DTYPE PATCH: training fed the model bf16 pixels - the
        # dataset/collator emit fp32 pixel_values and no explicit cast
        # exists in train.py, the vision tower was bf16 (train.py line
        # 921 under --bf16), and the bf16 projector (line 941) survived
        # all 8496 training forwards. The frozen run_generate passes
        # image_tensor.half() (fp16 pixels) and the OpenCLIP vision tower
        # emits features on the input-pixel dtype path -> fp16 features
        # into the bf16 projector = the v3_smoke matmul crash. Cast the
        # images kwarg to bf16 at the generate boundary so eval mirrors
        # training exactly. Instance attr shadows the class generate;
        # PeftModel.generate passes images= through to the base model
        # (proven by the smoke traceback). Frozen file stays untouched.
        _orig_generate = model.generate

        def _generate_bf16(*args, **kwargs):
            _imgs = kwargs.get("images", None)
            if _imgs is not None and torch.is_tensor(_imgs):
                kwargs["images"] = _imgs.to(torch.bfloat16)
            return _orig_generate(*args, **kwargs)

        model.generate = _generate_bf16
        print("  generate() wrapped: images kwarg cast to bf16 (matches "
              "training pixel dtype)", flush=True)

        vt = model.get_vision_tower()
        if not vt.is_loaded:
            print("loading BiomedCLIP-CXR vision tower (frozen, bf16) ...",
                  flush=True)
            vt.load_model()
        vt.to(device=device, dtype=torch.bfloat16)
        for i in range(torch.cuda.device_count()):
            print("  gpu%d: %.2f GB allocated"
                  % (i, torch.cuda.memory_allocated(i) / 1e9), flush=True)
        print("model ready (bf16)", flush=True)
        return model, tokenizer, vt.image_processor

    return load_model_bf16


def patch_ev(here, tag):
    """Import the frozen module (after the sha check) and apply all
    v3 patches. Returns the module."""
    check_frozen(here)
    global ev
    import eval_llavarad as ev_mod
    ev = ev_mod
    ev.FT_FILE = os.path.join(ev.IU, "finetuned_preds_%s.jsonl" % tag)
    ev.METRICS_JSON = os.path.join(ev.IU, "metrics_%s.json" % tag)
    ev.SHORTLIST = os.path.join(ev.IU, "review_shortlist_%s.jsonl" % tag)
    ev.load_model = make_bf16_load_model()
    print("preds   -> " + ev.FT_FILE, flush=True)
    print("metrics -> " + ev.METRICS_JSON, flush=True)
    print("shortlist -> " + ev.SHORTLIST, flush=True)
    return ev


def rewrite_notes(tag, max_new_tokens):
    """The frozen run_score writes a notes string that says '4-bit nf4
    double-quant'. Rewrite it for the v3 model (the file itself was
    produced by frozen code; only this human-readable string changes)."""
    with open(ev.METRICS_JSON) as f:
        m = json.load(f)
    m["notes"] = ("generation: greedy, max_new_tokens=%d, conv v1; model: bf16 "
                  "(no quantization) + LoRA r=64 unmerged, BiomedCLIP-CXR "
                  "frozen; rouge_l: whitespace-token LCS, case-sensitive"
                  % max_new_tokens)
    with open(ev.METRICS_JSON, "w") as f:
        json.dump(m, f, indent=2)
    print("notes rewritten (bf16, no quantization) in " + ev.METRICS_JSON,
          flush=True)


def main():
    ap = argparse.ArgumentParser(
        description="v3 (bf16) generate + score wrapper around frozen eval_llavarad.py")
    ap.add_argument("--tag", required=True,
                    help="output tag, e.g. v3_531 or v3_final")
    ap.add_argument("--adapter-dir", required=True,
                    help="dir with adapter_model.safetensors + non_lora_trainables.bin")
    ap.add_argument("--limit", type=int, default=0,
                    help="generate only: max samples this run (0 = all)")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--skip-generate", action="store_true",
                    help="score an existing preds file only")
    args = ap.parse_args()
    args.adapter_dir = os.path.expanduser(args.adapter_dir)

    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    REPO = os.path.join(os.path.expanduser("~"), "LLaVA-Rad")
    if REPO not in sys.path:
        sys.path.insert(0, REPO)

    patch_ev(here, args.tag)

    if not os.path.exists(os.path.join(args.adapter_dir,
                                       "adapter_model.safetensors")):
        raise SystemExit("ERROR: no adapter_model.safetensors in %s "
                         "(has training finished?)" % args.adapter_dir)
    if not args.skip_generate and not os.path.exists(
            os.path.join(args.adapter_dir, "non_lora_trainables.bin")):
        raise SystemExit("ERROR: no non_lora_trainables.bin in %s "
                         "(not the v3 model)" % args.adapter_dir)
    if args.skip_generate and not os.path.exists(ev.FT_FILE):
        raise SystemExit("ERROR: %s not found -- run generate first "
                         "(drop --skip-generate)" % ev.FT_FILE)

    fargs = argparse.Namespace(
        adapter_dir=args.adapter_dir, limit=args.limit,
        max_new_tokens=args.max_new_tokens, rg_reward="all",
        rg_model="radgraph-xl")

    test_map, base_map = ev.load_data()
    print("test set: %d samples; baseline file: %s"
          % (len(test_map), ev.BASE_FILE), flush=True)

    if not args.skip_generate:
        ev.run_generate(fargs, test_map, base_map)

    base_block = ev.compute_scores(ev.load_jsonl(ev.BASE_FILE), "baseline", fargs)
    ev.run_score(fargs, base_block, test_map, base_map)
    rewrite_notes(args.tag, args.max_new_tokens)

    import eval_extra as ex
    exargs = argparse.Namespace(clean=False, rg_model="radgraph-xl",
                                rg_cuda=None)
    ex.FINDINGS_JSON = "findings_metrics_%s.json" % args.tag
    ex.ENTITY_JSON = "entity_metrics_%s.json" % args.tag
    ex.run_findings(exargs)
    ex.run_entities(exargs)

    ex.FINDINGS_JSON = "findings_metrics_%sclean.json" % args.tag
    ex.ENTITY_JSON = "entity_metrics_%sclean.json" % args.tag
    exargs.clean = True
    ex.run_findings(exargs)
    ex.run_entities(exargs)

    print("", flush=True)
    print("DONE tag=%s  files:" % args.tag, flush=True)
    for fn in ("finetuned_preds_%s.jsonl" % args.tag,
               "metrics_%s.json" % args.tag,
               "review_shortlist_%s.jsonl" % args.tag,
               "findings_metrics_%s.json" % args.tag,
               "entity_metrics_%s.json" % args.tag,
               "findings_metrics_%sclean.json" % args.tag,
               "entity_metrics_%sclean.json" % args.tag):
        print("  " + os.path.join(ev.IU, fn), flush=True)


# module-global ev, set by patch_ev()
ev = None

if __name__ == "__main__":
    main()
