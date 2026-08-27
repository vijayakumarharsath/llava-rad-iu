#!/usr/bin/env python3
# cxr_show.py -- LLaVA-Rad (IU chest X-ray) terminal viewer + live demo
#
#   1  browse the saved 500 results (reference vs baseline vs finetuned)
#   2  step through the 16-sample review set (with scores)
#   3  live predict with the FINE-TUNED model (loaded once, ~5 GB VRAM)
#   4  show the metrics table
#
# No new dependencies: stdlib only. The live branch imports the VERIFIED
# load path from eval_llavarad.py (same 4-bit nf4 + unmerged LoRA + fp16
# projector code that produced the eval results). The generation block is
# copied verbatim from eval_llavarad.py run_generate(), which copied the
# official llava/eval/model_mimic_cxr.py.
#
# THIS TOOL SHOWS DRAFT REPORTS FOR HUMAN REVIEW. IT IS NEVER A DIAGNOSIS.

import json
import os
import sys
import time

HOME = os.path.expanduser("~")
IU = os.path.join(HOME, "iu-xray")
TRAIN_LOG = os.path.join(HOME, "train_resume3.log")
sys.path.insert(0, IU)
import eval_llavarad as ev  # verified eval script: load_model, rouge_l_f1, paths

W = 74
DEFAULT_QUERY = "Provide a description of the findings in the radiology image"
LIVE = {}  # {"model", "tokenizer", "iproc"} -- set once by _live_load()


def hr(char="-"):
    print(char * W)


def ask(prompt):
    try:
        return input(prompt)
    except (EOFError, KeyboardInterrupt):
        print()
        return "q"


def load_all():
    """Load the data files once. No GPU, no model."""
    with open(ev.TEST_JSON) as f:
        test = json.load(f)
    data = {
        "test_ids": [r["id"] for r in test],
        "test_map": {r["id"]: r for r in test},
        "base": {r["id"]: r for r in ev.load_jsonl(ev.BASE_FILE)},
        "ft": {},
        "short": [],
    }
    if os.path.exists(ev.FT_FILE):
        data["ft"] = {r["id"]: r for r in ev.load_jsonl(ev.FT_FILE)}
    if os.path.exists(ev.SHORTLIST):
        data["short"] = ev.load_jsonl(ev.SHORTLIST)
    return data


def show_block(title, text):
    print("-- %s --" % title)
    print(str(text).strip())
    print()


def show_case(rid, data, live_pred=None):
    """Show one case: reference, saved baseline, saved finetuned, live one."""
    t = data["test_map"].get(rid, {})
    b = data["base"].get(rid)
    f = data["ft"].get(rid)
    print()
    print("=" * W)
    print(rid + (("   image: " + t["image"]) if t.get("image") else ""))
    hr()
    if b:
        show_block("query", b["query"].replace("<image>", "").strip())
        show_block("reference (radiologist)", b["reference"])
        show_block("baseline (official LLaVA-Rad)", b["prediction"])
    if f:
        show_block("finetuned (your LoRA, saved)", f["prediction"])
    if live_pred is not None:
        show_block("finetuned (LIVE, just now)", live_pred[0])
    if b and (f or live_pred):
        parts = ["ROUGE-L vs reference:  baseline=%.3f"
                 % ev.rouge_l_f1(b["prediction"], b["reference"])]
        if f:
            parts.append("finetuned=%.3f"
                         % ev.rouge_l_f1(f["prediction"], b["reference"]))
        if live_pred is not None:
            parts.append("live=%.3f"
                         % ev.rouge_l_f1(live_pred[0], b["reference"]))
        print("  ".join(parts))
    for s in data["short"]:
        if s["id"] != rid:
            continue
        print("[review set] tag=%s" % s.get("reason", ""))
        for k in sorted(s):
            if k.startswith(("baseline_", "finetuned_", "delta_")) \
                    and isinstance(s[k], (int, float)):
                print("  %-22s %.4f" % (k, s[k]))
    print()


def resolve_id(s, data):
    """Resolve a full id, a 1-500 number, or a unique prefix (e.g. 'CXR2516').
    Returns (rid, None) on success, else (None, candidate_list)."""
    if s in data["test_map"]:
        return s, None
    if s.isdigit():
        n = int(s)
        if 1 <= n <= len(data["test_ids"]):
            return data["test_ids"][n - 1], None
    cands = [i for i in data["test_ids"] if i.lower().startswith(s.lower())]
    if len(cands) == 1:
        return cands[0], None
    return None, cands


def _print_cands(s, cands):
    if cands:
        print("no unique id for %s -- %d matches, e.g.:" % (s, len(cands)))
        for c in cands[:10]:
            print("  " + c)
        if len(cands) > 10:
            print("  ... and %d more (type more characters)" % (len(cands) - 10))
    else:
        print("not found: %s  (ids: use 'list' or a prefix like CXR2516;"
              " files: give a full path to a .png/.dcm file)" % s)


def cmd_browse(data):
    ids = data["test_ids"]
    print("saved results: %d baseline, %d finetuned, %d in the review set"
          % (len(data["base"]), len(data["ft"]), len(data["short"])))
    page = 0
    while True:
        s = ask("id / prefix (e.g. CXR2516) / number 1-%d / 'list' / q: "
                % len(ids)).strip()
        if s.lower() in ("", "q"):
            return
        if s.lower() in ("list", "l"):
            lo = page * 50
            in_review = {x["id"] for x in data["short"]}
            for i, rid in enumerate(ids[lo:lo + 50], lo + 1):
                mark = "*" if rid in in_review else " "
                print("%4d%s %s" % (i, mark, rid))
            print("( * = in the 16-sample review set; 'list' again = next page )")
            page += 1
            continue
        rid, cands = resolve_id(s, data)
        if rid is None:
            _print_cands(s, cands)
            continue
        page = 0
        show_case(rid, data)


def cmd_review(data):
    if not data["short"]:
        print("no review_shortlist.jsonl found in %s" % IU)
        return
    print("stepping through the %d review samples" % len(data["short"]))
    i = 0
    while True:
        show_case(data["short"][i]["id"], data)
        k = ask("  [%d/%d] Enter/n = next, p = prev, id/prefix = jump, q = back: "
                % (i + 1, len(data["short"]))).strip().lower()
        if k in ("q", "quit"):
            return
        if k in ("", "n"):
            i = (i + 1) % len(data["short"])
        elif k == "p":
            i = (i - 1) % len(data["short"])
        else:
            match = [j for j, x in enumerate(data["short"])
                     if x["id"].lower() == k or x["id"].lower().startswith(k)]
            if len(match) == 1:
                i = match[0]
            elif match:
                print("  %d review ids start with %s; type more characters"
                      % (len(match), k))
            else:
                print("  (not a review id -- Enter = next, p = prev, q = back)")


def _live_load():
    if LIVE:
        return
    print("loading the FINE-TUNED model once (~2 min, ~5 GB VRAM) ...")
    if ev.REPO not in sys.path:
        sys.path.insert(0, ev.REPO)
    t0 = time.time()
    LIVE["model"], LIVE["tokenizer"], LIVE["iproc"] = ev.load_model(
        ev.ADAPTER_DIR, "cuda")
    print("ready in %.0fs -- the model stays loaded; re-entering menu "
          "item 3 is instant." % (time.time() - t0))


def _open_image(path):
    """PNG/JPG straight from disk; DICOM (.dcm or extensionless) decoded in
    memory with the SAME recipe as the training pipeline: apply
    RescaleSlope/Intercept, clip to a chest window, resize to 518x518.
    Returns a PIL image."""
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    if path.lower().endswith((".png", ".jpg", ".jpeg")):
        return Image.open(path).convert("RGB")
    import pydicom
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array
    if arr.ndim == 3:  # multi-frame: take the largest frame
        sizes = [int(f.size) for f in arr]
        arr = arr[sizes.index(max(sizes))]
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    v = arr.astype("float32") * slope + intercept
    if str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")) == "MONOCHROME1":
        v = v.max() + v.min() - v
    lo, hi = -500.0, 250.0  # chest window
    g = ((v - lo) * (255.0 / (hi - lo))).clip(0, 255)
    if float(g.max() - g.min()) < 5:  # flat -> plain min-max
        vmin, vmax = float(v.min()), float(v.max())
        g = ((v - vmin) / (vmax - vmin + 1e-6) * 255.0).clip(0, 255)
    lz = getattr(getattr(Image, "Resampling", Image), "LANCZOS")  # old+new Pillow
    return (Image.fromarray(g.astype("uint8"), mode="L")
            .resize((518, 518), lz).convert("RGB"))


def _predict(model, tokenizer, iproc, image, query):
    """One greedy 256-token report. Verbatim mechanics from the official eval
    (via eval_llavarad.py run_generate): conv v1, prompt = <image>\\n + query,
    images fp16 on cuda, stop at '###' or EOS. `image` is a PIL image."""
    import torch
    from llava.conversation import conv_templates, SeparatorStyle
    from llava.mm_utils import tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

    cv0 = conv_templates["v1"]
    stop_str = cv0.sep if cv0.sep_style != SeparatorStyle.TWO else cv0.sep2
    q = query.replace("<image>", "").strip()
    q = DEFAULT_IMAGE_TOKEN + "\n" + q
    conv = cv0.copy()
    conv.append_message(conv.roles[0], q)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                      return_tensors="pt")  # 1D (seq_len,)
    image_tensor = iproc.preprocess(image.convert("RGB"),
                                    return_tensors="pt")["pixel_values"][0]
    t0 = time.time()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids.unsqueeze(0).to("cuda"),
            images=image_tensor.half().unsqueeze(0).to("cuda"),
            do_sample=False, temperature=0, num_beams=1,
            max_new_tokens=256, use_cache=True).cpu()
    dt = time.time() - t0
    out_ids = output_ids[0][input_ids.shape[0]:].tolist()
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id in out_ids:
        out_ids = out_ids[:out_ids.index(tokenizer.eos_token_id)]
    pred = tokenizer.decode(out_ids, skip_special_tokens=True).strip()
    if pred.endswith(stop_str):
        pred = pred[:-len(stop_str)].strip()
    return pred, dt


def cmd_live(data):
    try:
        _live_load()
    except Exception as e:
        print("model load failed: %s" % e)
        return
    model, tok, iproc = LIVE["model"], LIVE["tokenizer"], LIVE["iproc"]
    print("type a test id or prefix (CXR2516), or a path to a png/dcm file;")
    print("q = back to menu")
    while True:
        s = ask("id / prefix / path to png or dcm (q = back): ").strip()
        if s.lower() in ("q", "quit", ""):
            return
        rid_r, cands = resolve_id(s, data)
        image_path = None
        rid, query = None, DEFAULT_QUERY
        if rid_r is not None:
            rid = rid_r
            image_path = os.path.join(ev.IMAGE_DIR, data["test_map"][rid_r]["image"])
            if rid_r in data["base"]:
                query = data["base"][rid_r]["query"]
        elif os.path.exists(s):
            image_path = s
        elif os.path.exists(os.path.join(ev.IMAGE_DIR, s)):
            image_path = os.path.join(ev.IMAGE_DIR, s)
        if image_path is None:
            _print_cands(s, cands)
            continue
        try:
            image = _open_image(image_path)
            pred, dt = _predict(model, tok, iproc, image, query)
        except Exception as e:
            print("predict failed: %s" % e)
            continue
        title = rid if rid else "custom: " + os.path.basename(image_path)
        show_case(title, data, live_pred=(pred, dt))
        print("(live prediction took %.1fs)" % dt)


def cmd_metrics(data):
    if not os.path.exists(ev.METRICS_JSON):
        print("no metrics.json yet (run the eval first)")
        return
    with open(ev.METRICS_JSON) as f:
        m = json.load(f)
    head = m.get("headline_metric", "rg_er")
    b, ft = m.get("baseline", {}), m.get("finetuned", {})
    brg, frg = b.get("rg", {}), ft.get("rg", {})
    print("headline metric: %s (closest to the original 0.287)" % head)
    print()
    print("%-10s %-8s %-8s %-10s %-9s" % ("", "rg_e", "rg_er", "rg_bar_er", "rouge_l"))
    for name, blk in (("baseline", b), ("finetuned", ft)):
        rg = blk.get("rg", {})
        print("%-10s %-8.4f %-8.4f %-10.4f %-9.4f"
              % (name, rg.get("rg_e", 0.0), rg.get("rg_er", 0.0),
                 rg.get("rg_bar_er", 0.0), blk.get("rouge_l", 0.0)))
    print("%-10s % +8.4f % +8.4f % +10.4f % +9.4f"
          % ("delta", frg.get("rg_e", 0.0) - brg.get("rg_e", 0.0),
             frg.get("rg_er", 0.0) - brg.get("rg_er", 0.0),
             frg.get("rg_bar_er", 0.0) - brg.get("rg_bar_er", 0.0),
             ft.get("rouge_l", 0.0) - b.get("rouge_l", 0.0)))
    print()
    print("delta %s (finetuned - baseline): %+.4f   n=%s"
          % (head, m.get("delta_headline", 0.0), b.get("n", "?")))
    if m.get("notes"):
        print("notes: " + m["notes"])


def cmd_train(data):
    print()
    print("=" * W)
    print("FINE-TUNING DETAILS (LLaVA-Rad + QLoRA, real run)")
    print("=" * W)
    rows = [
        ("base model", "LLaVA-Rad: Vicuna-7B LLM + BiomedCLIP-CXR 518 encoder"),
        ("method", "QLoRA: 4-bit NF4 double-quant + LoRA r=64 / alpha=128 / dropout 0.05"),
        ("trained", "LoRA adapters (q,k,v,o,gate,up,down) + image->text projector"),
        ("frozen", "vision encoder + base LLM"),
        ("data", "2837 IU X-ray image-report pairs (train split, per patient)"),
        ("epochs", "3"),
        ("iterations", "531 total = 177 per epoch (2837 / effective batch 16)"),
        ("batch", "effective 16 (1 image x 16 gradient accumulation, 1 GPU)"),
        ("lr", "1e-4, cosine schedule, 3% warmup, weight decay 0"),
        ("precision", "bf16 + TF32, gradient checkpointing, max seq len 2048"),
        ("speed", "~51 s per iteration"),
        ("final run", "231 iterations (step 300 -> 531) in 11800 s = ~3 h 17 m"),
        ("final loss", "0.212 at epoch 3.0"),
    ]
    for k, v in rows:
        print("%-13s %s" % (k + ":", v))
    if os.path.exists(TRAIN_LOG):
        line = ""
        with open(TRAIN_LOG, errors="replace") as f:
            for ln in f:
                if "train_runtime" in ln:
                    line = ln.strip()
        if line:
            print()
            print("real log line (" + os.path.basename(TRAIN_LOG) + "):")
            print("  " + line)
    print()


def main():
    print("=" * W)
    print("LLaVA-Rad (IU chest X-ray) -- terminal viewer + live demo")
    print("DRAFT REPORTS FOR HUMAN REVIEW -- NEVER A DIAGNOSIS")
    print("=" * W)
    try:
        data = load_all()
    except Exception as e:
        raise SystemExit("ERROR loading data from %s: %s" % (IU, e))
    while True:
        print()
        print("  1  browse saved results (500 test X-rays)")
        print("  2  review set (16 samples, with scores)")
        print("  3  live predict with the finetuned model")
        print("  4  metrics table")
        print("  5  training info (epochs, iterations, run details)")
        print("  q  quit")
        c = ask("choice: ").strip().lower()
        if c == "1":
            cmd_browse(data)
        elif c == "2":
            cmd_review(data)
        elif c == "3":
            cmd_live(data)
        elif c == "4":
            cmd_metrics(data)
        elif c == "5":
            cmd_train(data)
        elif c in ("q", "quit", "exit"):
            print("bye.")
            return
        else:
            print("(pick 1 / 2 / 3 / 4 / q)")


if __name__ == "__main__":
    main()
