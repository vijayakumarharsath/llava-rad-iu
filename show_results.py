#!/usr/bin/env python3
# show_results.py -- print the full evaluation table from the saved result
# JSONs. No model load, no GPU, no re-scoring: reads only the versioned
# result files in ~/iu-xray and prints one table. Safe to run as often as
# you like; output is deterministic.
#
#   ~/llavarad-env/bin/python ~/iu-xray/show_results.py

import json
import os
import re

IU = os.path.expanduser("~/iu-xray")


def load(name):
    p = os.path.join(IU, name)
    if not os.path.exists(p):
        raise SystemExit("missing " + p)
    with open(p) as f:
        return json.load(f)


def ray_count_text(text):
    # same counting rule as rayclean.py: standalone 'ray' tokens,
    # case-insensitive, hyphenated x-ray never counts
    return sum(1 for t in re.findall(r"[\w'-]+", text) if t.lower() == "ray")


def ray_count_file(fname, field):
    n = 0
    with open(os.path.join(IU, fname)) as f:
        for line in f:
            line = line.strip()
            if line:
                n += ray_count_text(json.loads(line).get(field, ""))
    return n


def main():
    m1 = load("metrics_v1.json")          # v1 run:  baseline + v1
    m2 = load("metrics_v2.json")          # v2 run:  baseline + v2
    m3a = load("metrics_v3_531.json")     # v3 run:  3ep checkpoint (best)
    m3b = load("metrics_v3_final.json")   # v3 run:  6ep final checkpoint
    e1 = load("entity_metrics_v1.json")   # entity FP/FN, published refs
    e2 = load("entity_metrics_v2.json")
    e3a = load("entity_metrics_v3_531.json")
    e3b = load("entity_metrics_v3_final.json")
    f1 = load("findings_metrics_v1.json") # 14 CheXbert labels
    f2 = load("findings_metrics_v2.json")
    f3a = load("findings_metrics_v3_531.json")
    f3b = load("findings_metrics_v3_final.json")

    base, v1, v2 = (m1["baseline"]["rg"], m1["finetuned"]["rg"],
                    m2["finetuned"]["rg"])
    v3a, v3b = m3a["finetuned"]["rg"], m3b["finetuned"]["rg"]
    a1b = e1["baseline"]["entities"]["abnormal"]
    a1f = e1["finetuned"]["entities"]["abnormal"]
    a2f = e2["finetuned"]["entities"]["abnormal"]
    a3af = e3a["finetuned"]["entities"]["abnormal"]
    a3bf = e3b["finetuned"]["entities"]["abnormal"]

    rows = [
        ("RadGraph rg_e",            [base["rg_e"], v1["rg_e"], v2["rg_e"],
                                      v3a["rg_e"], v3b["rg_e"]], 4),
        ("RadGraph rg_er (headline)", [base["rg_er"], v1["rg_er"],
                                       v2["rg_er"], v3a["rg_er"],
                                       v3b["rg_er"]], 4),
        ("RadGraph rg_bar_er",       [base["rg_bar_er"], v1["rg_bar_er"],
                                      v2["rg_bar_er"], v3a["rg_bar_er"],
                                      v3b["rg_bar_er"]], 4),
        ("ROUGE-L F1",               [m1["baseline"]["rouge_l"],
                                      m1["finetuned"]["rouge_l"],
                                      m2["finetuned"]["rouge_l"],
                                      m3a["finetuned"]["rouge_l"],
                                      m3b["finetuned"]["rouge_l"]], 4),
        ("CheXbert-14 macro F1",     [f1["macro_f1"]["baseline"],
                                      f1["macro_f1"]["finetuned"],
                                      f2["macro_f1"]["finetuned"],
                                      f3a["macro_f1"]["finetuned"],
                                      f3b["macro_f1"]["finetuned"]], 4),
        ("entity FP/report (abn)",   [a1b["fp_per_report"],
                                      a1f["fp_per_report"],
                                      a2f["fp_per_report"],
                                      a3af["fp_per_report"],
                                      a3bf["fp_per_report"]], 3),
        ("entity FN/report (abn)",   [a1b["fn_per_report"],
                                      a1f["fn_per_report"],
                                      a2f["fn_per_report"],
                                      a3af["fn_per_report"],
                                      a3bf["fn_per_report"]], 3),
        ("missed findings (abn)",    [f1["split"]["baseline"]["abn_fn"],
                                      f1["split"]["finetuned"]["abn_fn"],
                                      f2["split"]["finetuned"]["abn_fn"],
                                      f3a["split"]["finetuned"]["abn_fn"],
                                      f3b["split"]["finetuned"]["abn_fn"]], 0),
        ("false findings (abn)",     [f1["split"]["baseline"]["abn_fp"],
                                      f1["split"]["finetuned"]["abn_fp"],
                                      f2["split"]["finetuned"]["abn_fp"],
                                      f3a["split"]["finetuned"]["abn_fp"],
                                      f3b["split"]["finetuned"]["abn_fp"]], 0),
        ("false alarms (normal)",    [f1["split"]["baseline"]["normal_false_alarm"],
                                      f1["split"]["finetuned"]["normal_false_alarm"],
                                      f2["split"]["finetuned"]["normal_false_alarm"],
                                      f3a["split"]["finetuned"]["normal_false_alarm"],
                                      f3b["split"]["finetuned"]["normal_false_alarm"]], 0),
        ("'ray' artifact tokens",    [ray_count_file("baseline_preds.jsonl",
                                                     "prediction"),
                                      ray_count_file("finetuned_preds_v1.jsonl",
                                                     "prediction"),
                                      ray_count_file("finetuned_preds_v2.jsonl",
                                                     "prediction"),
                                      ray_count_file("finetuned_preds_v3_531.jsonl",
                                                     "prediction"),
                                      ray_count_file("finetuned_preds_v3_final.jsonl",
                                                     "prediction")], 0),
    ]

    print("LLaVA-Rad fine-tuned on IU chest X-rays -- evaluation, n=%d test"
          % m1["baseline"]["n"])
    print()
    print("%-25s | %-9s | %-9s | %-9s | %-9s | %-9s"
          % ("metric", "baseline", "v1", "v2", "v3 3ep", "v3 6ep"))
    print("-" * 82)
    for name, vals, nd in rows:
        line = "%-25s | " % name
        line += " | ".join(("%9." + str(nd) + "f") % v for v in vals)
        print(line)
    print()
    print("v1 = QLoRA fine-tune (4-bit, 3ep, single GPU)")
    print("v2 = v1 recipe on ray-cleaned + abnormal-rebalanced data (experiment)")
    print("v3 = full bf16, no quantization, 6ep, 2x GPU ZeRO-3;")
    print("     'v3 3ep' = that run's epoch-3 checkpoint (best, taken to MIMIC)")
    print("RadGraph = clinical-fact graph match (rg_er: entities + relation)")
    print("abn = abnormal references, normal = normal references")
    print("labels = 14 CheXbert conditions via rule labeler (findlabel.py)")
    print("baseline reproduced exactly before trusting the pipeline")


if __name__ == "__main__":
    main()
