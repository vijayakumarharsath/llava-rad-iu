#!/usr/bin/env python3
# mimic_score.py -- score the MIMIC-CXR baseline vs finetuned prediction
# files with the FROZEN project pipeline (no new metric code).
#
# Why: reuse eval_extra.py's findings axis (14 CheXbert conditions via
# findlabel.py, CPU, seconds) and entities axis (radgraph-xl entity
# FP/FN + rg_e/rg_er/rg_bar_er + ROUGE-L, GPU, a few minutes) exactly as
# they were used for the IU rounds, by redirecting the module's file
# constants to the MIMIC files -- the same monkeypatch pattern
# run_v3_eval.py already proved (ev.FT_FILE / ev.METRICS_JSON /
# ev.SHORTLIST). eval_llavarad.py is imported for load_jsonl + constants,
# its frozen sha is verified first, and it is never modified.
#
# The two prediction files must be the official worker's output
# (baseline) and mimic_eval.py's output (finetuned): same schema
# {"id","query","reference","prediction"}, same ids. Scoring proceeds on
# the id intersection (a warning prints if they differ).
#
# Outputs (in --out_dir):
#   findings_metrics_<tag>.json   (same schema as the IU findings files)
#   entity_metrics_<tag>.json     (same schema as the IU entity files,
#                                  incl. rg + rouge_l blocks)
# plus a short headline summary on stdout for the paste-back.

import argparse
import json
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

FROZEN_SHA = "0b88eaaea85daef74901fa7e16873942aaaaffbb686167bb544b07aa389a9ad9"


def check_frozen():
    import hashlib
    h = hashlib.sha256(open(os.path.join(HERE, "eval_llavarad.py"),
                           "rb").read()).hexdigest()
    if h != FROZEN_SHA:
        sys.exit("FATAL: eval_llavarad.py sha %s != frozen %s"
                 % (h[:16], FROZEN_SHA[:16]))
    print("eval_llavarad.py frozen sha OK", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_file", required=True)
    ap.add_argument("--finetuned_file", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tag", default="mimic")
    ap.add_argument("--mode", choices=["findings", "entities", "all"],
                    default="all")
    ap.add_argument("--rg-cuda", type=int, default=None,
                    help="pin radgraph to this GPU (default: auto)")
    args = ap.parse_args()
    check_frozen()

    for f in (args.baseline_file, args.finetuned_file):
        if not os.path.exists(f):
            sys.exit("missing " + f)
    os.makedirs(args.out_dir, exist_ok=True)

    import eval_llavarad as ev
    import eval_extra as xx

    # --- redirect the frozen pipeline to the MIMIC files -----------------
    ev.BASE_FILE = args.baseline_file
    ev.FT_FILE = args.finetuned_file
    ev.IU = args.out_dir
    # must NOT exist: run_entities compares against it as a sanity check
    ev.METRICS_JSON = os.path.join(args.out_dir, "metrics_%s.json"
                                   % args.tag)
    xx.FINDINGS_JSON = "findings_metrics_%s.json" % args.tag
    xx.ENTITY_JSON = "entity_metrics_%s.json" % args.tag

    b = ev.load_jsonl(ev.BASE_FILE)
    ft = ev.load_jsonl(ev.FT_FILE)
    bset = set(r["id"] for r in b)
    fset = set(r["id"] for r in ft)
    shared = len(bset & fset)
    print("baseline n=%d  finetuned n=%d  shared=%d"
          % (len(bset), len(fset), shared), flush=True)
    if shared < min(len(bset), len(fset)):
        print("WARNING: %d ids not shared; scoring proceeds on the "
              "intersection" % (min(len(bset), len(fset)) - shared),
              file=sys.stderr)
    if shared == 0:
        sys.exit("FATAL: no shared ids between the two prediction files")

    ns = types.SimpleNamespace(clean=False, rg_model="radgraph-xl",
                               rg_cuda=args.rg_cuda)
    if args.mode in ("findings", "all"):
        print("\n== FINDINGS (14 CheXbert conditions, rule labeler) ==",
              flush=True)
        xx.run_findings(ns)
    if args.mode in ("entities", "all"):
        print("\n== ENTITIES (radgraph-xl) + RadGraph + ROUGE-L ==",
              flush=True)
        xx.run_entities(ns)

    # --- headline summary -------------------------------------------------
    # each block guarded by file existence so --mode findings / entities
    # work standalone (with --mode all both files exist)
    e_path = os.path.join(args.out_dir, xx.ENTITY_JSON)
    f_path = os.path.join(args.out_dir, xx.FINDINGS_JSON)
    print("\n== MIMIC RESULTS (headline metric: rg_er) ==")
    if os.path.exists(e_path):
        e = json.load(open(e_path))
        for name in ("baseline", "finetuned"):
            r, m = e[name]["rg"], e[name]["rouge_l"]
            a = e[name]["entities"]["abnormal"]
            print("%-9s rg_e=%.4f rg_er=%.4f rg_bar_er=%.4f rouge_l=%.4f"
                  % (name, r["rg_e"], r["rg_er"], r["rg_bar_er"], m))
            print("%-9s ent abn FP/rep=%.3f FN/rep=%.3f prec=%.4f rec=%.4f"
                  % (name, a["fp_per_report"], a["fn_per_report"],
                     a["entity_precision"], a["entity_recall"]))
        db = e["baseline"]["rg"]["rg_er"]
        df = e["finetuned"]["rg"]["rg_er"]
        print("delta rg_er (finetuned - baseline): %+.4f" % (df - db))
    if os.path.exists(f_path):
        f = json.load(open(f_path))
        print("findings macro F1: baseline=%.4f finetuned=%.4f  (n=%d, "
              "refs abn=%d normal=%d)"
              % (f["macro_f1"]["baseline"], f["macro_f1"]["finetuned"], f["n"],
                 f["refs_abnormal"], f["refs_normal"]))
        fs = f["split"]["finetuned"]
        print("finetuned split: abn FP=%d abn FN=%d normal false alarms=%d"
              % (fs["abn_fp"], fs["abn_fn"], fs["normal_false_alarm"]))
    print("SCORING DONE tag=%s" % args.tag)


if __name__ == "__main__":
    main()
