#!/usr/bin/env python3
# mimic_subset.py -- slice the LLaVA-Rad MIMIC-CXR annotation json into a
# deterministic subset for the external-validation run.
#
# Why: the official test split (loader 'mimic_test_findings' in
# microsoft/LLaVA-Rad llava/utils.py) is the paper's eval set. We run the
# FIRST N of it (deterministic json order) with BOTH the official llavarad
# model (baseline) and the v3 3ep checkpoint, so the comparison is fully
# matched. The filter below is copied VERBATIM from the official
# data_loader_mimic_reason_findings(..., "test") so the subset is exactly
# what the official loader keeps (the worker re-applies the same idempotent
# transform, so feeding it this file changes nothing).
#
# Output: json list (same schema as the source), first N filtered entries.
# Prints: total entries, post-filter count, number written, id range.
# No GPU, no model, safe to run repeatedly.

import argparse
import json


def filter_test(entries):
    # VERBATIM from llava/utils.py data_loader_mimic_reason_findings (test)
    ret = []
    for d in entries:
        if d["generate_method"] != "rule-based":
            continue
        # Skip empty findings
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True,
                    help="full annotation json (chat_test_...json)")
    ap.add_argument("--n", type=int, default=2000,
                    help="subset size (default 2000)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data = json.load(open(args.json))
    kept = filter_test(data)
    sub = kept[:args.n]
    with open(args.out, "w") as f:
        json.dump(sub, f)
    print("total entries in json      : %d" % len(data))
    print("after official test filter : %d" % len(kept))
    print("written to subset          : %d -> %s" % (len(sub), args.out))
    if sub:
        print("first id: %s   last id: %s"
              % (sub[0].get("id"), sub[-1].get("id")))
        print("sample image path: %s" % sub[0]["image"])


if __name__ == "__main__":
    main()
