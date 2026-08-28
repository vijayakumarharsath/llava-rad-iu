#!/usr/bin/env python3
# findlabel.py -- deterministic rule-based labeler for the 14 CheXbert
# conditions (negation-aware, sentence-level).
#
# WHY: we need a finding-level evaluation axis (normal/abnormal split +
# per-label F1) with zero new dependencies. The label list is the OFFICIAL
# CheXbert one (stanfordmlgroup/CheXbert, src/constants.py, verified from
# source 2026-08-28):
#
#   CONDITIONS = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
#                 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
#                 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
#                 'Pleural Other', 'Fracture', 'Support Devices', 'No Finding']
#
# NOTE: the official pretrained CheXbert weights (HF repo
# stanfordmlgroup/chexbert) are no longer publicly available as of
# 2026-08-28, so this deterministic approximation is the labeler we ship.
# It is documented as an approximation in the write-up:
#   - sentence = split on . ! ? and newlines
#   - a phrase match is NEGATED if a negator token (no / not / without /
#     cannot / can't / negative) sits within the previous 8 tokens of the
#     match, OR within the next 4 tokens ("effusion is not seen")
#   - a report is ABNORMAL if any pathology condition fires positive, or if
#     an extended abnormality keyword list fires (emphysema, fibrosis,
#     infiltrate, hyperinflation, bullae, aspiration, ...). The extended
#     list is ONLY used for the normal/abnormal split, never scored in the
#     14-condition F1 table. The 14 conditions deliberately have no
#     infiltrate/emphysema/fibrosis entries (that is the CheXpert set);
#     reports dominated by those findings contribute little to this axis,
#     which the RadGraph entity metric covers.
#
# Usage:
#   python3 findlabel.py "no evidence of pulmonary edema"
#   python3 findlabel.py --show "mild cardiomegaly is present"
#   python3 findlabel.py --jsonl ~/iu-xray/baseline_preds.jsonl --field reference
#   python3 findlabel.py --selftest
#
# Importable: from findlabel import label_text, LABELS

import argparse
import json
import re
import sys
from collections import Counter

LABELS = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
          'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
          'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
          'Support Devices', 'No Finding']

# phrase lists per condition; a trailing "*" marks a stem (no trailing word
# boundary). Matching is case-insensitive, word-bounded at the front.
PHRASES = {
    'Enlarged Cardiomediastinum': [
        "enlarged cardiomediastinum",
        "cardiomediastinal silhouette is enlarged",
        "cardiomediastinal silhouette enlargement",
        "widened mediastinum", "mediastinal widening",
        "widened mediastinal silhouette", "mediastinum is widened",
        "enlarged mediastinum"],
    'Cardiomegaly': [
        "cardiomegaly", "cardiomegalic", "enlarged heart",
        "heart is enlarged", "heart appears enlarged",
        "cardiac silhouette is enlarged", "enlarged cardiac silhouette"],
    'Lung Opacity': ["opacity", "opacities", "opacification"],
    'Lung Lesion': ["mass", "masses", "nodule", "nodules", "nodular",
                    "nodularity", "lesion", "lesions"],
    'Edema': ["edema"],
    'Consolidation': ["consolidation", "consolidations"],
    'Pneumonia': ["pneumonia"],
    'Atelectasis': ["atelectasis", "atelectatic"],
    'Pneumothorax': ["pneumothorax", "pneumothoraces"],
    'Pleural Effusion': ["effusion", "effusions"],
    'Pleural Other': ["pleural thickening", "pleural scarring",
                      "pleural irregularit*", "apical irregularit*",
                      "pleural-parenchymal irregularit*"],
    'Fracture': ["fracture", "fractures"],
    'Support Devices': ["catheter", "pacemaker", "endotracheal",
                        "nasogastric", "enteric tube", "enteral tube",
                        "picc", "central venous", "support device",
                        "support devices", "dialysis catheter", "stent",
                        "stents", "tubes", "tubing", "line", "lines"],
}

# phrases that mark an explicitly normal report (whole-text substring)
NO_FINDING_PHRASES = [
    "no acute cardiopulmonary", "no acute abnormality", "no acute process",
    "no acute intrathoracic", "no acute finding", "no acute findings",
    "no significant abnormality", "unremarkable", "within normal limits",
    "normal chest", "chest is normal", "lungs are clear", "clear lungs",
    "no focal airspace opacity", "xray is normal", "x-ray is normal",
]

# split-only extra abnormality keywords (substring, case-insensitive)
EXT_ABNORMAL = [
    "emphysema", "emphysematous", "fibrosis", "fibrotic", "infiltrat",
    "hyperinflat", "hyperexpand", "hyperlucent", "bulla", "bullae",
    "bullous", "aspiration", "pneumoconiosis",
]

# negators before the match: "no"/"not"/... within PRE_WINDOW tokens
PRE_TOKENS = {"no", "not", "without", "cannot", "can't", "negative"}
PRE_WINDOW = 8
# negators after the match: "edema is not seen", "atelectasis cannot be
# excluded". "no" is deliberately NOT here: it is a sentence-initial
# operator, and post-match "no" usually starts the next clause
# ("edema. no radiographic evidence pneumonia").
POST_TOKENS = {"not", "without", "cannot", "can't", "negative"}
POST_WINDOW = 4

_SENT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_TOK_RE = re.compile(r"[\w'-]+")


def _phrase_rx(phrase):
    stem = phrase.endswith("*")
    core = phrase.rstrip("*")
    rx = r"\b" + re.escape(core)
    if not stem:
        rx += r"\b"
    return rx, core


def _negated(pre, post):
    ptoks = _TOK_RE.findall(pre)
    if any(t in PRE_TOKENS for t in ptoks[-PRE_WINDOW:]):
        return True
    ntoks = _TOK_RE.findall(post)
    return any(t in POST_TOKENS for t in ntoks[:POST_WINDOW])


def label_text(text):
    """Label one report. Returns
    {"labels": {14 conditions: 0/1}, "abnormal": bool,
     "abnormal_reasons": [...], "details": [(label, phrase, negated, sent)]}"""
    labels = {l: 0 for l in LABELS}
    details = []
    if not text or not text.strip():
        return {"labels": labels, "abnormal": False,
                "abnormal_reasons": [], "details": details}

    any_path = False
    for s in _SENT_RE.split(text):
        for lab in LABELS:
            if lab == "No Finding":
                continue
            for phrase in PHRASES[lab]:
                rx, core = _phrase_rx(phrase)
                for m in re.finditer(rx, s, flags=re.IGNORECASE):
                    neg = _negated(s[:m.start()], s[m.end():])
                    details.append((lab, core, neg, s.strip()))
                    if not neg:
                        labels[lab] = 1
                        any_path = True

    low = text.lower()
    labels["No Finding"] = 1 if (any(p in low for p in NO_FINDING_PHRASES)
                                 and not any_path) else 0

    ext = sorted(p for p in EXT_ABNORMAL if p in low)
    abnormal = any_path or bool(ext)
    reasons = [l for l in LABELS if l != "No Finding" and labels[l]] + ext
    return {"labels": labels, "abnormal": abnormal,
            "abnormal_reasons": reasons, "details": details}


def is_abnormal(res):
    return res["abnormal"]


def selftest():
    def L(s):
        return label_text(s)["labels"]

    def ABN(s):
        return label_text(s)["abnormal"]

    # negation cases
    assert L("no evidence of pulmonary edema")["Edema"] == 0
    lab = L("unchanged mild pulmonary edema no radiographic evidence pneumonia")
    assert lab["Edema"] == 1 and lab["Pneumonia"] == 0, lab
    assert L("no focal alveolar consolidation")["Consolidation"] == 0
    lab = L("There is no pleural effusion or pneumothorax.")
    assert lab["Pleural Effusion"] == 0 and lab["Pneumothorax"] == 0, lab
    assert L("no cardiomegaly")["Cardiomegaly"] == 0
    assert L("cardiomegaly is not present")["Cardiomegaly"] == 0
    assert L("atelectasis cannot be excluded")["Atelectasis"] == 0
    assert L("cannot rule out pneumonia")["Pneumonia"] == 0
    assert L("negative for effusion")["Pleural Effusion"] == 0
    assert L("effusion is not seen")["Pleural Effusion"] == 0
    assert L("no rib fracture identified")["Fracture"] == 0
    # positive cases
    assert L("mild cardiomegaly is present")["Cardiomegaly"] == 1
    lab = L("the heart is enlarged and there is no edema")
    assert lab["Cardiomegaly"] == 1 and lab["Edema"] == 0, lab
    assert L("right middle lobe infiltrate with mild cardiomegaly")["Cardiomegaly"] == 1
    assert L("moderate pleural effusion")["Pleural Effusion"] == 1
    assert L("endotracheal tube terminates above the carina")["Support Devices"] == 1
    assert L("a 6 mm nodule is seen in the left upper lobe")["Lung Lesion"] == 1
    assert L("focal airspace opacity in the right lower lobe")["Lung Opacity"] == 1
    assert L("apical pleural-parenchymal irregularities compatible with scarring")["Pleural Other"] == 1
    assert L("widened mediastinum")["Enlarged Cardiomediastinum"] == 1
    # house-style normal (Ishita demo verbatim)
    house = ("The heart is normal in size. The mediastinum is unremarkable. "
             "There is no pleural effusion or pneumothorax. There is no focal "
             "airspace opacity to suggest a pneumonia.")
    lab = L(house)
    assert lab["No Finding"] == 1 and not ABN(house), lab
    # abnormality flag
    assert ABN("no acute cardiopulmonary abnormality") is False
    assert ABN("moderate emphysema with hyperinflation") is True  # ext list
    assert label_text("moderate emphysema")["labels"]["Lung Opacity"] == 0
    # empty
    assert label_text("")["abnormal"] is False
    print("findlabel selftest: all OK")


def main():
    ap = argparse.ArgumentParser(description="CheXbert-14 rule labeler")
    ap.add_argument("text", nargs="?", help="report text to label")
    ap.add_argument("--show", help="label this text and print sentence details")
    ap.add_argument("--jsonl", help="label every record's --field in a jsonl file")
    ap.add_argument("--field", default="reference")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if args.show:
        res = label_text(args.show)
        print(json.dumps(res["labels"], indent=1))
        print("abnormal:", res["abnormal"], "reasons:", res["abnormal_reasons"])
        for lab, core, neg, sent in res["details"]:
            print("  %-26s %-28s %-8s :: %s" % (lab, core,
                                                "NEG" if neg else "POS",
                                                sent[:70]))
        return
    if args.text:
        print(json.dumps(label_text(args.text), indent=1))
        return
    if args.jsonl:
        recs = []
        with open(args.jsonl) as f:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
        res = [label_text(r.get(args.field, "")) for r in recs]
        freq = Counter()
        for r in res:
            for l in LABELS:
                if r["labels"][l]:
                    freq[l] += 1
        print("n=%d  abnormal=%d  normal=%d"
              % (len(res), sum(1 for r in res if r["abnormal"]),
                 sum(1 for r in res if not r["abnormal"])))
        for l, n in freq.most_common(14):
            print("%4d  %s" % (n, l))
        return
    ap.print_help()


if __name__ == "__main__":
    main()
