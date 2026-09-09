#!/usr/bin/env python3
# mimic_eval.py -- generate MIMIC-CXR reports with the v3 3ep bf16
# IU fine-tuned checkpoint (checkpoint-531), on the same query file and
# image folder as the official LLaVA-Rad worker, so the external
# validation comparison is fully matched.
#
# Why / what differs from the official worker (all documented, none
# change the comparison):
#   * model = OUR bf16 model: base LLM bf16 (no quantization), fine-tuned
#     projector from non_lora_trainables.bin, LoRA loaded UNMERGED via
#     PeftModel, vision tower bf16, and pixels cast to bf16 at the
#     generate boundary -- the exact dtype mix v3 trained with (derivation
#     in patch_run_v3_eval_dtype.py; this is the same loader run_v3_eval.py
#     uses for the IU evals).
#   * batch 1 (IU eval memory/speed figure is per-sample at batch 1).
#   * append-safe: rerun skips ids already in the prediction file, so a
#     crash or laptop-off never loses progress.
#   * DONE tag + progress every 50 samples (same launch pattern as the
#     IU evals: setsid nohup ... &).
#
# What matches the official worker (the comparison-critical parts):
#   * data: the official 'mimic_test_findings' filter (copied VERBATIM
#     from llava/utils.py; idempotent, so an already-sliced file is
#     unchanged)
#   * conv template v1, prompt = IMAGE_TOKEN + "\n" + query (no im
#     start/end tokens, cfg.mm_use_im_start_end = False)
#   * greedy decoding (do_sample=False, temperature 0, num_beams 1),
#     max_new_tokens 256 -- the official eval.sh passes --temperature 0
#   * output line schema {"id","query","reference","prediction"}
#
# eval_llavarad.py is imported for its path constants only and its
# frozen sha is verified first; it is never modified.

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

FROZEN_SHA = "0b88eaaea85daef74901fa7e16873942aaaaffbb686167bb544b07aa389a9ad9"


def check_frozen():
    import hashlib
    p = os.path.join(HERE, "eval_llavarad.py")
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    if h != FROZEN_SHA:
        sys.exit("FATAL: eval_llavarad.py sha %s != frozen %s"
                 % (h[:16], FROZEN_SHA[:16]))
    print("eval_llavarad.py frozen sha OK", flush=True)


def load_mimic_test(path):
    # VERBATIM from microsoft/LLaVA-Rad llava/utils.py
    # data_loader_mimic_reason_findings(data_path, "test"). Idempotent:
    # re-applying to an already-sliced/transformed file changes nothing.
    with open(path) as f:
        dataset = json.load(f)
    ret = []
    for d in dataset:
        if d["generate_method"] != "rule-based":
            continue
        if not isinstance(d["conversations"][1]["value"], str):
            continue
        if d["view"] not in ("AP", "PA"):
            continue
        if d["image"].startswith("mimic/"):
            d["image"] = d["image"][len("mimic/"):]
        if d["reason"] is not None:
            reason = d["reason"].replace("\n", " ")
            d["conversations"][0]["value"] = (
                "<image>\nProvide a description of the findings in the "
                "radiology image given the following indication: " + reason)
        else:
            d["conversations"][0]["value"] = (
                "<image>\nProvide a description of the findings in the "
                "radiology image.")
        ret.append(d)
    return ret


def done_ids(path):
    s = set()
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line:
                s.add(json.loads(line)["id"])
    return s


def load_model_bf16(adapter_dir, device, ev):
    # Port of run_v3_eval.py load_model_bf16 WITH the dtype patch
    # (vision bf16 + generate wrapper casting the images kwarg to bf16).
    import torch
    from transformers import AutoConfig, AutoTokenizer
    from peft import PeftModel
    from llava.model import LlavaLlamaForCausalLM  # registers "llava"

    cfg_file = os.path.join(adapter_dir, "config.json")
    if os.path.exists(cfg_file):
        cfg = AutoConfig.from_pretrained(adapter_dir)
        if getattr(cfg, "quantization_config", None) is not None:
            cfg.quantization_config = None
            print("note: stripped embedded quantization_config", flush=True)
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
              "from the base", flush=True)
    # Vision routing: same values training used.
    cfg.mm_vision_tower = "biomedclip_cxr_518"
    cfg.mm_vision_tower_config = ev.VIT_CONFIG
    cfg.mm_vision_tower_checkpoint = ev.VIT_CKPT

    tokenizer = AutoTokenizer.from_pretrained(ev.MODEL_BASE, use_fast=False)
    print("loading base LLM (bf16, NO quantization) ...", flush=True)
    model = LlavaLlamaForCausalLM.from_pretrained(
        ev.MODEL_BASE, low_cpu_mem_usage=True, config=cfg,
        device_map="auto", torch_dtype=torch.bfloat16)
    model.resize_token_embeddings(len(tokenizer))

    non_lora_file = os.path.join(adapter_dir, "non_lora_trainables.bin")
    if not os.path.exists(non_lora_file):
        sys.exit("ERROR: no non_lora_trainables.bin in %s (refusing to "
                 "fall back to the pre-training projector)" % adapter_dir)
    proj_sd = torch.load(non_lora_file, map_location="cpu",
                         weights_only=False)
    # same key normalization as the official builder (model/builder.py)
    proj_sd = {(k[11:] if k.startswith("base_model.") else k): v
               for k, v in proj_sd.items()}
    if any(k.startswith("model.model.") for k in proj_sd):
        proj_sd = {(k[6:] if k.startswith("model.") else k): v
                   for k, v in proj_sd.items()}
    dense = {}
    for k, v in proj_sd.items():
        if "mm_projector" in k:
            dense[".".join(k.split(".")[-2:])] = v
    if set(dense) != {"0.weight", "0.bias", "2.weight", "2.bias"}:
        sys.exit("ERROR: unexpected projector keys: %s" % sorted(dense))
    import torch.nn as nn
    proj = nn.Sequential(
        nn.Linear(dense["0.weight"].shape[1], model.config.hidden_size),
        nn.GELU(),
        nn.Linear(model.config.hidden_size, model.config.hidden_size))
    missing, unexpected = proj.load_state_dict(dense)
    if missing or unexpected:
        sys.exit("projector mismatch: missing=%s unexpected=%s"
                 % (missing, unexpected))
    model.model.mm_projector = proj.to(device="cuda", dtype=torch.bfloat16)
    print("  projector installed (bf16)", flush=True)

    print("loading LoRA adapter: " + adapter_dir, flush=True)
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    # dtype patch: training fed the model bf16 pixels (vision tower bf16,
    # bf16 projector). Cast the images kwarg to bf16 at the generate
    # boundary so eval mirrors training exactly.
    _orig_generate = model.generate

    def _generate_bf16(*args, **kwargs):
        _imgs = kwargs.get("images", None)
        if _imgs is not None and torch.is_tensor(_imgs):
            kwargs["images"] = _imgs.to(torch.bfloat16)
        return _orig_generate(*args, **kwargs)

    model.generate = _generate_bf16
    print("  generate() wrapped: images kwarg cast to bf16", flush=True)

    vt = model.get_vision_tower()
    if not vt.is_loaded:
        print("loading BiomedCLIP-CXR vision tower (frozen, bf16) ...",
              flush=True)
        vt.load_model()
    vt.to(device=device, dtype=torch.bfloat16)
    print("model ready (bf16)", flush=True)
    return model, tokenizer, vt.image_processor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query_file", required=True)
    ap.add_argument("--image_folder", required=True)
    ap.add_argument("--adapter_dir", default=os.path.expanduser(
        "~/llava-rad-checkpoints/llava-rad-iu-v3/checkpoint-531"))
    ap.add_argument("--pred_file", required=True)
    ap.add_argument("--tag", default="mimic")
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = all; else first N of loader order")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    args = ap.parse_args()

    check_frozen()
    import eval_llavarad as ev
    import torch
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    from llava.conversation import conv_templates, SeparatorStyle
    from llava.mm_utils import tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

    queries = load_mimic_test(args.query_file)
    if args.limit > 0:
        queries = queries[:args.limit]
    done = done_ids(args.pred_file)
    todo = [d for d in queries if d["id"] not in done]
    print("queries=%d done=%d todo=%d -> %s"
          % (len(queries), len(done), len(todo), args.pred_file), flush=True)
    if not todo:
        print("DONE tag=%s (nothing to do)" % args.tag, flush=True)
        return

    device = "cuda"
    model, tokenizer, image_processor = load_model_bf16(
        args.adapter_dir, device, ev)
    cv0 = conv_templates["v1"]
    stop_str = cv0.sep if cv0.sep_style != SeparatorStyle.TWO else cv0.sep2

    t0 = time.time()
    ddir = os.path.dirname(os.path.abspath(args.pred_file))
    if ddir:
        os.makedirs(ddir, exist_ok=True)
    with open(args.pred_file, "a") as out:
        for i, d in enumerate(todo, 1):
            q = d["conversations"][0]["value"].replace("<image>", "").strip()
            q = DEFAULT_IMAGE_TOKEN + "\n" + q
            ref = d["conversations"][1]["value"]
            conv = cv0.copy()
            conv.append_message(cv0.roles[0], q)
            conv.append_message(cv0.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            image = Image.open(
                os.path.join(args.image_folder, d["image"])).convert("RGB")
            image_tensor = image_processor.preprocess(
                image, return_tensors="pt")["pixel_values"][0]
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids.unsqueeze(0).to(device),
                    # unsqueeze(0): batch dim is REQUIRED -- the model indexes
                    # image_features[batch_idx] (llava_arch.py). Frozen loop
                    # passes .half().unsqueeze(0); here fp32 + the generate
                    # wrapper casts to bf16.
                    images=image_tensor.unsqueeze(0).to(device),
                    do_sample=False, temperature=0, num_beams=1,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True).cpu()
            out_ids = output_ids[0][input_ids.shape[0]:].tolist()
            if tokenizer.eos_token_id is not None \
                    and tokenizer.eos_token_id in out_ids:
                out_ids = out_ids[:out_ids.index(tokenizer.eos_token_id)]
            pred = tokenizer.decode(out_ids, skip_special_tokens=True).strip()
            if pred.endswith(stop_str):
                pred = pred[:-len(stop_str)].strip()
            out.write(json.dumps({"id": d["id"], "query": q,
                                  "reference": ref,
                                  "prediction": pred}) + "\n")
            out.flush()
            if i % 50 == 0 or i == len(todo):
                el = time.time() - t0
                print("[%d/%d] avg=%.2fs eta=%.1fmin last_id=%s"
                      % (i, len(todo), el / i,
                         el / i * (len(todo) - i) / 60, d["id"]), flush=True)
    print("DONE tag=%s n=%d elapsed=%.1fmin file=%s"
          % (args.tag, len(todo), (time.time() - t0) / 60,
             args.pred_file), flush=True)


if __name__ == "__main__":
    main()
