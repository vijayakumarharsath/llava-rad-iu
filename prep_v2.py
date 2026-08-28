#!/usr/bin/env python3
# prep_v2.py -- build the v2 training data: ray-cleaned reports + abnormal
# oversampling weights.
#
# WHY: v1 trained on 2,837 image-report pairs. The house template makes
# normal studies the majority, and the reports contain "ray" artifacts that
# the model learned to repeat. v2 keeps the EXACT SAME 2,837 pairs (same
# ids, same images, same patient split -- so the 500-patient test comparison
# stays clean) and changes only two things:
#   1. the gpt reports are ray-cleaned (rayclean.py, auditable rules)
#   2. per-sample weights: abnormal x2 (default) / normal x1. The weights
#      are consumed by a WeightedRandomSampler patch in train.py, which
#      keeps the iteration count unchanged: the sampler draws
#      len(dataset) samples per epoch, so v2 runs the same ~177
#      iters/epoch as v1 (no 9-hour duplication run).
#
# Outputs (in ~/iu-xray): train_v2.json, v2_weights.json.
# train.json is NEVER touched.
#
# Usage:
#   python3 prep_v2.py --report          # ray contexts + label split on train refs
#   python3 prep_v2.py --build           # build with abnormal weight 2.0
#   python3 prep_v2.py --build --weight 3.0
#   python3 prep_v2.py --selftest

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rayclean import clean_ray, ray_contexts
from findlabel import label_text

IU = os.path.expanduser(os.environ.get("IU_DIR", "~/iu-xray"))
TRAIN_JSON = os.path.join(IU, "llava_iu", "train.json")


def _gpt(rec):
    for c in rec["conversations"]:
        if c.get("from") == "gpt" or "gpt" in c:
            return c
    raise SystemExit("record has no gpt turn: %s" % rec.get("id"))


def load_train():
    with open(TRAIN_JSON) as f:
        return json.load(f)


def report(data):
    ctx = Counter()
    changed = 0
    n_abn = 0
    for rec in data:
        g = _gpt(rec)["value"]
        for key, _case in ray_contexts(g):
            ctx[key] += 1
        c, ch = clean_ray(g)
        changed += sum(n for _, n in ch)
        if label_text(c)["abnormal"]:
            n_abn += 1
    n = len(data)
    print("train pairs: %d" % n)
    print("abnormal refs (findlabel): %d (%.1f%%)" % (n_abn, 100.0 * n_abn / n))
    print("ray occurrences: %d ; current RAY_RULES would change %d places"
          % (sum(ctx.values()), changed))
    print("top 25 ray contexts in train refs:")
    for key, c in ctx.most_common(25):
        print("%4d  %s" % (c, key))


def build(data, weight):
    n = len(data)
    out = []
    weights = []
    abn_ids = []
    n_cleaned = 0
    samples = []
    for rec in data:
        g = _gpt(rec)
        cleaned, ch = clean_ray(g["value"])
        nch = sum(c for _, c in ch)
        if nch:
            n_cleaned += 1
            if len(samples) < 3:
                samples.append((rec["id"], g["value"], cleaned))
        r2 = json.loads(json.dumps(rec))
        _gpt(r2)["value"] = cleaned
        out.append(r2)
        if label_text(cleaned)["abnormal"]:
            weights.append(weight)
            abn_ids.append(rec["id"])
        else:
            weights.append(1.0)

    n_abn = len(abn_ids)
    total_w = n_abn * weight + (n - n_abn) * 1.0
    print("pairs: %d (unchanged)" % n)
    print("ray-cleaned pairs: %d" % n_cleaned)
    print("abnormal: %d (%.1f%%)   normal: %d (%.1f%%)"
          % (n_abn, 100.0 * n_abn / n, n - n_abn, 100.0 * (n - n_abn) / n))
    print("oversample weight: %.1f (abnormal), 1.0 (normal)" % weight)
    print("effective abnormal fraction: %.1f%% (was %.1f%%)"
          % (100.0 * n_abn * weight / total_w, 100.0 * n_abn / n))
    print("sampler draws %d samples/epoch -> ~177 iters/epoch at batch 16 "
          "(same count as v1)" % n)
    if samples:
        print("example changes (first 3):")
        for rid, old, new in samples:
            print("  %s:" % rid)
            print("    old: %s" % old[:130].replace("\n", " "))
            print("    new: %s" % new[:130].replace("\n", " "))
    with open(os.path.join(IU, "train_v2.json"), "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(IU, "v2_weights.json"), "w") as f:
        json.dump({"n": n, "weight_abnormal": weight,
                   "abnormal_ids": abn_ids, "weights": weights}, f)
    print("wrote train_v2.json + v2_weights.json (train.json untouched)")


def selftest():
    data = [
        {"id": "T1", "image": "t1.png",
         "conversations": [{"from": "human", "value": "<image>\nDescribe."},
                           {"from": "gpt", "value": "no acute process on this x-ray"}]},
        {"id": "T2", "image": "t2.png",
         "conversations": [{"from": "human", "value": "<image>\nDescribe."},
                           {"from": "gpt", "value": "atherosclerotic calcifications of the aortic ray are noted"}]},
        {"id": "T3", "image": "t3.png",
         "conversations": [{"from": "human", "value": "<image>\nDescribe."},
                           {"from": "gpt", "value": "moderate cardiomegaly is present"}]},
    ]
    c1 = clean_ray(data[0]["conversations"][-1]["value"])
    c2 = clean_ray(data[1]["conversations"][-1]["value"])
    assert c1[0] == data[0]["conversations"][-1]["value"] and c1[1] == []
    assert c2[0].endswith("of the aortic arch are noted"), c2
    abn = [label_text(_gpt(r)["value"])["abnormal"] for r in data]
    assert abn == [False, False, True], abn
    print("prep_v2 selftest: all OK")


def main():
    ap = argparse.ArgumentParser(description="build v2 training data")
    ap.add_argument("--report", action="store_true",
                    help="ray contexts + label split on train refs (no writes)")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--weight", type=float, default=2.0,
                    help="oversample weight for abnormal refs (default 2.0)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    data = load_train()
    if args.report:
        report(data)
    elif args.build:
        build(data, args.weight)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
