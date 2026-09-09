#!/usr/bin/env python3
# eval_extra.py -- extra evaluation axes for the LLaVA-Rad IU project.
#
# Built 08-28 in answer to mentor feedback ("the model is hallucinating" +
# "use other evaluation metrics"). This file imports eval_llavarad.py for
# paths + ROUGE + jsonl loading and NEVER modifies it (v7 stays frozen).
#
# Modes:
#   smoke     2-pair radgraph-xl probe. Step 0 on the machine: proves the
#             radgraph path works end to end and shows what the entity
#             sets look like before we trust the big runs.
#   findings  Finding-level axis: the 14 CheXbert conditions via the
#             rule labeler (findlabel.py). Per-label TP/FP/FN + macro F1
#             for baseline and finetuned, plus the normal/abnormal split
#             (false alarms on normal refs, omissions on abnormal refs).
#             CPU only, seconds. --clean applies rayclean to refs and
#             preds before labeling (keep this flag consistent across
#             rows of the final table).
#   entities  Entity-level axis -- the direct answer to "does it
#             hallucinate": radgraph-xl entity sets, where
#               FP = entities in PRED not in REF  (made-up / ungrounded)
#               FN = entities in REF not in PRED  (omitted)
#             aggregated overall and in the normal/abnormal split, plus
#             the same rg_e/rg_er/rg_bar_er + ROUGE-L on exactly these
#             pairs (without --clean they must reproduce metrics.json,
#             which double-checks the plumbing). GPU, a few minutes.
#             NOTE: an "ungrounded" entity (FP) is not necessarily a false
#             finding -- the report can contain a true-but-undocumented
#             finding. That nuance goes in the write-up.
#   all       findings + entities.
#
# Run order (agreed plan):
#   0  python3 eval_extra.py smoke              (2 min)
#   1  python3 rayclean.py --report             (seconds; may extend rules)
#   2  python3 eval_extra.py findings
#   3  python3 eval_extra.py entities           (uncleaned v1 rows)
#   4  python3 eval_extra.py entities --clean   (ray-cleaned v1 rows)
#   5  (later) findings --clean for the cleaned table row, if needed

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_llavarad as ev
from rayclean import clean_ray
from findlabel import label_text, LABELS

FINDINGS_JSON = "findings_metrics.json"
ENTITY_JSON = "entity_metrics.json"
PATH_LABELS = [l for l in LABELS if l != "No Finding"]


def _load_pairs():
    """(id, reference, baseline_pred, finetuned_pred) aligned on baseline."""
    base = ev.load_jsonl(ev.BASE_FILE)
    ft = {r["id"]: r for r in ev.load_jsonl(ev.FT_FILE)}
    missing = [r["id"] for r in base if r["id"] not in ft]
    if missing:
        print("WARNING: %d baseline ids missing from finetuned file, e.g. %s"
              % (len(missing), missing[0]), file=sys.stderr)
    pairs = [(r["id"], r["reference"], r["prediction"], ft[r["id"]]["prediction"])
             for r in base if r["id"] in ft]
    return pairs


def _clean_pair(pair, args, counters):
    if not args.clean:
        return pair
    rid, ref, pb, pf = pair
    ref, ch = clean_ray(ref);  counters["refs"] += sum(n for _, n in ch)
    pb, ch = clean_ray(pb);    counters["base"] += sum(n for _, n in ch)
    pf, ch = clean_ray(pf);    counters["ft"] += sum(n for _, n in ch)
    return (rid, ref, pb, pf)


# ------------------------------- smoke ----------------------------------


def run_smoke(args):
    from radgraph import F1RadGraph
    kw = {"model_type": args.rg_model}
    if args.rg_cuda is not None:
        kw["cuda"] = args.rg_cuda
    print("loading radgraph-xl (already cached from the eval runs) ...",
          flush=True)
    t0 = time.time()
    metric = F1RadGraph(reward_level="all", **kw)
    print("loaded in %.1fs" % (time.time() - t0), flush=True)
    refs = ["no acute cardiopulmonary abnormality",
            "right middle lobe infiltrate with mild cardiomegaly"]
    hyps = ["no acute cardiopulmonary abnormality",
            "there is a right middle lobe infiltrate and the heart is enlarged"]
    means, per, hyp_anns, ref_anns = metric(refs=refs, hyps=hyps)
    assert len(hyp_anns) == 2 and len(ref_anns) == 2
    for i in range(2):
        h, r = hyp_anns[i]["entities"], ref_anns[i]["entities"]
        print("case %d: %d hyp entities, %d ref entities" % (i + 1, len(h), len(r)))
        for e in list(h.values())[:3]:
            print("   hyp: %-38s [%s]" % (e["tokens"], e["label"]))
        for e in list(r.values())[:3]:
            print("   ref: %-38s [%s]" % (e["tokens"], e["label"]))
    print("mean rg_e=%.4f rg_er=%.4f rg_bar_er=%.4f"
          % (float(means[0]), float(means[1]), float(means[2])))
    print("SMOKE-OK")


# ------------------------------ findings --------------------------------


def _f1(tp, fp, fn):
    return 2.0 * tp / (2.0 * tp + fp + fn) if (tp + fp + fn) else 0.0


def run_findings(args):
    counters = {"refs": 0, "base": 0, "ft": 0}
    pairs = [_clean_pair(p, args, counters) for p in _load_pairs()]
    n = len(pairs)
    if args.clean:
        print("rayclean applied: refs=%d base=%d ft=%d"
              % (counters["refs"], counters["base"], counters["ft"]), flush=True)
    ref_res = [label_text(p[1]) for p in pairs]
    base_res = [label_text(p[2]) for p in pairs]
    ft_res = [label_text(p[3]) for p in pairs]
    n_abn = sum(1 for r in ref_res if r["abnormal"])

    stats = {name: {l: [0, 0, 0] for l in LABELS}
             for name in ("baseline", "finetuned")}  # [tp, fp, fn]
    split = {name: {"abn_fp": 0, "abn_fn": 0, "normal_false_alarm": 0}
             for name in ("baseline", "finetuned")}
    for i in range(n):
        rgt = ref_res[i]["labels"]
        ref_abn = ref_res[i]["abnormal"]
        for name, res in (("baseline", base_res[i]["labels"]),
                          ("finetuned", ft_res[i]["labels"])):
            for lab in LABELS:
                if res[lab] and rgt[lab]:
                    stats[name][lab][0] += 1
                elif res[lab]:
                    stats[name][lab][1] += 1
                elif rgt[lab]:
                    stats[name][lab][2] += 1
            pred_pos = [l for l in PATH_LABELS if res[l]]
            ref_pos = [l for l in PATH_LABELS if rgt[l]]
            if ref_abn:
                split[name]["abn_fp"] += len(set(pred_pos) - set(ref_pos))
                split[name]["abn_fn"] += len(set(ref_pos) - set(pred_pos))
            elif pred_pos:
                split[name]["normal_false_alarm"] += 1

    out = {"clean": bool(args.clean), "n": n,
           "refs_normal": n - n_abn, "refs_abnormal": n_abn,
           "labeler": ("rule-based (findlabel.py), 14 CheXbert conditions, "
                       "negation-aware; official CheXbert weights not "
                       "publicly available (HF repo gone as of 2026-08-28)"),
           "per_label": {}, "macro_f1": {}, "split": split}
    print("refs: normal=%d abnormal=%d  (14 CheXbert conditions, rule labeler%s)"
          % (n - n_abn, n_abn, ", ray-cleaned" if args.clean else ""))
    print("%-26s | %-19s | %-19s" % ("label", "base TP FP FN F1",
                                     "ft   TP FP FN F1"))
    macro = {"baseline": 0.0, "finetuned": 0.0}
    for lab in LABELS:
        b = stats["baseline"][lab]
        f = stats["finetuned"][lab]
        fb, ff = _f1(*b), _f1(*f)
        macro["baseline"] += fb
        macro["finetuned"] += ff
        out["per_label"][lab] = {
            "baseline": {"tp": b[0], "fp": b[1], "fn": b[2], "f1": round(fb, 4)},
            "finetuned": {"tp": f[0], "fp": f[1], "fn": f[2], "f1": round(ff, 4)},
        }
        print("%-26s | %3d %3d %3d %.3f | %3d %3d %3d %.3f"
              % (lab, b[0], b[1], b[2], fb, f[0], f[1], f[2], ff))
    macro = {k: round(v / len(LABELS), 4) for k, v in macro.items()}
    out["macro_f1"] = macro
    print("%-26s | macro F1 = %.4f | macro F1 = %.4f"
          % ("", macro["baseline"], macro["finetuned"]))
    for name in ("baseline", "finetuned"):
        s = split[name]
        print("%s: FP on abnormal refs=%d  FN on abnormal refs=%d  "
              "false alarms on normal refs=%d"
              % (name, s["abn_fp"], s["abn_fn"], s["normal_false_alarm"]))
    out_path = os.path.join(ev.IU, FINDINGS_JSON)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print("findings -> " + out_path)


# ------------------------------ entities --------------------------------


def _entity_agg(entries):
    n = len(entries)
    tp = sum(e[0] for e in entries)
    fp = sum(e[1] for e in entries)
    fn = sum(e[2] for e in entries)
    return {
        "n": n, "tp": tp, "fp": fp, "fn": fn,
        "fp_per_report": round(fp / n, 4) if n else 0.0,
        "fn_per_report": round(fn / n, 4) if n else 0.0,
        "entity_precision": round(tp / (tp + fp), 4) if (tp + fp) else 0.0,
        "entity_recall": round(tp / (tp + fn), 4) if (tp + fn) else 0.0,
    }


def _per_case(hyps, refs, hyp_anns, ref_anns):
    """(tp, fp, fn) per case, aligned with the radgraph annotation lists
    (which skip pairs where either side is an empty string -- same rule
    radgraph uses internally). None for skipped pairs."""
    nonempty = [len(h) > 0 and len(r) > 0 for h, r in zip(hyps, refs)]
    res = []
    k = 0
    for i in range(len(refs)):
        if not nonempty[i]:
            res.append(None)
            continue
        h = hyp_anns[k]["entities"]
        r = ref_anns[k]["entities"]
        k += 1
        hs = {(e["tokens"], e["label"]) for e in h.values()}
        rs = {(e["tokens"], e["label"]) for e in r.values()}
        res.append((len(hs & rs), len(hs - rs), len(rs - hs)))
    assert k == sum(nonempty), "annotation alignment mismatch"
    return res


def run_entities(args):
    counters = {"refs": 0, "base": 0, "ft": 0}
    pairs = [_clean_pair(p, args, counters) for p in _load_pairs()]
    ids = [p[0] for p in pairs]
    refs = [p[1] for p in pairs]
    hyps_b = [p[2] for p in pairs]
    hyps_f = [p[3] for p in pairs]
    abn = [label_text(r)["abnormal"] for r in refs]
    if args.clean:
        print("rayclean applied: refs=%d base=%d ft=%d"
              % (counters["refs"], counters["base"], counters["ft"]), flush=True)
    print("n=%d  refs abnormal=%d normal=%d"
          % (len(refs), sum(abn), len(refs) - sum(abn)), flush=True)

    from radgraph import F1RadGraph
    kw = {"model_type": args.rg_model}
    if args.rg_cuda is not None:
        kw["cuda"] = args.rg_cuda
    t0 = time.time()
    print("loading radgraph-xl ...", flush=True)
    metric = F1RadGraph(reward_level="all", **kw)
    means_b, _, hyp_anns_b, ref_anns_b = metric(refs=refs, hyps=hyps_b)
    means_f, _, hyp_anns_f, ref_anns_f = metric(refs=refs, hyps=hyps_f)
    print("radgraph done in %.0fs (2 x %d pairs)" % (time.time() - t0, len(refs)),
          flush=True)

    cb = _per_case(hyps_b, refs, hyp_anns_b, ref_anns_b)
    cf = _per_case(hyps_f, refs, hyp_anns_f, ref_anns_f)
    rl_b = sum(ev.rouge_l_f1(hyps_b[i], refs[i]) for i in range(len(refs))) / len(refs)
    rl_f = sum(ev.rouge_l_f1(hyps_f[i], refs[i]) for i in range(len(refs))) / len(refs)

    def block(cases, means, rl):
        all_e = [c for c in cases if c is not None]
        ab_e = [c for c, a in zip(cases, abn) if c is not None and a]
        no_e = [c for c, a in zip(cases, abn) if c is not None and not a]
        return {
            "rg": {"rg_e": round(float(means[0]), 4),
                   "rg_er": round(float(means[1]), 4),
                   "rg_bar_er": round(float(means[2]), 4)},
            "rouge_l": round(rl, 4),
            "entities": {"overall": _entity_agg(all_e),
                         "abnormal": _entity_agg(ab_e),
                         "normal": _entity_agg(no_e)},
        }

    blk_b = block(cb, means_b, rl_b)
    blk_f = block(cf, means_f, rl_f)

    if not args.clean and os.path.exists(ev.METRICS_JSON):
        with open(ev.METRICS_JSON) as f:
            saved = json.load(f)
        ok = True
        for name, blk in (("baseline", blk_b), ("finetuned", blk_f)):
            s = saved.get(name, {})
            for k in ("rg_e", "rg_er", "rg_bar_er"):
                if k in s.get("rg", {}) and abs(s["rg"][k] - blk["rg"][k]) > 1e-4:
                    ok = False
                    print("SANITY-DRIFT %s %s: saved=%.4f here=%.4f"
                          % (name, k, s["rg"][k], blk["rg"][k]))
        print("sanity vs metrics.json: %s" % ("PASS (rg values match)"
              if ok else "DRIFT (inspect before trusting)"))

    print("")
    print("== ENTITY FP/FN  (FP = in pred not in ref, FN = in ref not in pred) ==")
    print("%-10s %-9s | n   TP   FP   FN  FP/rep FN/rep  ent-P  ent-R" % ("model", "split"))
    for name, blk in (("baseline", blk_b), ("finetuned", blk_f)):
        for split_name in ("overall", "abnormal", "normal"):
            e = blk["entities"][split_name]
            print("%-10s %-9s | %3d %4d %4d %4d  %.3f  %.3f  %.3f  %.3f"
                  % (name, split_name, e["n"], e["tp"], e["fp"], e["fn"],
                     e["fp_per_report"], e["fn_per_report"],
                     e["entity_precision"], e["entity_recall"]))
    for name, blk in (("baseline", blk_b), ("finetuned", blk_f)):
        print("%-10s rg_e=%.4f rg_er=%.4f rg_bar_er=%.4f rouge_l=%.4f%s"
              % (name, blk["rg"]["rg_e"], blk["rg"]["rg_er"],
                 blk["rg"]["rg_bar_er"], blk["rouge_l"],
                 "  (ray-cleaned)" if args.clean else ""))
    out_path = os.path.join(ev.IU, ENTITY_JSON)
    with open(out_path, "w") as f:
        json.dump({"clean": bool(args.clean),
                   "n": len(refs),
                   "refs_abnormal": sum(abn),
                   "note": ("FP = entity (tokens,label) in prediction not in "
                            "reference (ungrounded; not necessarily a false "
                            "finding). FN = reference entity missing from "
                            "prediction. Entity identity = the same "
                            "(tokens,label) key radgraph uses for rg_e."),
                   "baseline": blk_b, "finetuned": blk_f}, f, indent=2)
    print("entities -> " + out_path)


# -------------------------------- main ----------------------------------


def main():
    ap = argparse.ArgumentParser(description="extra eval axes (hallucination + findings)")
    ap.add_argument("mode_pos", nargs="?", default=None,
                    help="smoke | findings | entities | all (default: all)")
    ap.add_argument("--mode", dest="mode_flag",
                    choices=["smoke", "findings", "entities", "all"], default=None,
                    help="same as the positional mode (kept for compat)")
    ap.add_argument("--clean", action="store_true",
                    help="apply rayclean to refs and preds before scoring")
    ap.add_argument("--rg-model", default="radgraph-xl")
    ap.add_argument("--rg-cuda", type=int, default=None,
                    help="pin radgraph to this GPU (default: auto, usually 0)")
    args = ap.parse_args()
    mode = args.mode_pos or args.mode_flag or "all"
    if mode == "smoke":
        run_smoke(args)
    if mode in ("findings", "all"):
        run_findings(args)
    if mode in ("entities", "all"):
        run_entities(args)


if __name__ == "__main__":
    main()
