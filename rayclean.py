#!/usr/bin/env python3
# rayclean.py -- auditable cleanup of "ray" artifacts in IU report text.
#
# WHY: the IU house reports (and so the model trained on them) contain
# anonymization leftovers where the word "ray" sits in odd contexts
# ("atherosclerotic calcifications of the aortic ray", "ray sternotomy ray").
# The model then repeats them. This script fixes them with a small, AUDITABLE
# rule table:
#   1. ordered context rules (exact substrings, case-insensitive) -- the only
#      changes the DEFAULT mode ever makes
#   2. x-ray variants are protected with placeholders (never touched)
#   3. leftover standalone lowercase "ray" is removed ONLY with --aggressive
#      (and only then is the "no ray of X" trace idiom also protected)
#   4. text repair (double spaces, " ," -> ",", "the the" -> "the", ...)
#
# Nothing changes silently: `rayclean.py --report` prints every "…ray…"
# context found in the data with counts, so the rule table below can be
# extended from evidence. After the report phase, extend RAY_RULES if new
# contexts show up; the default (rules-only) mode never removes a "ray" it
# does not explicitly know.
#
# Usage:
#   python3 rayclean.py --report                # scan test refs + both pred files
#   python3 rayclean.py --text "aortic ray calcifications"
#   python3 rayclean.py --text "the ray arch" --aggressive
#   python3 rayclean.py --selftest
#
# Importable: from rayclean import clean_ray, ray_contexts, RAY_RULES

import argparse
import json
import os
import re
import sys
from collections import Counter

IU = os.path.expanduser(os.environ.get("IU_DIR", "~/iu-xray"))

# Ordered context rules: (pattern, replacement). First match wins per span.
# Extend from `--report` output. Every rule is visible here = auditable.
# (08-28 round 3: extended from --sentences on real data.)
RAY_RULES = [
    ("ray sternotomy ray", "sternotomy"),
    ("sternotomy ray", "sternotomy"),
    ("ray sternotomy", "sternotomy"),
    ("have ray ray in the interval", "have no change in the interval"),
    ("ray ray", "x-ray"),
    ("aortic ray", "aortic arch"),
    ("costophrenic ray", "costophrenic angle"),
    ("ray examination", "x-ray examination"),
    ("no ray of", "no signs of"),
    ("lymph ray", "lymph nodes"),
    ("pulmonary ray", "pulmonary vasculature"),
    ("ray are", "ribs are"),
]
# known residual (left untouched, documented): "Normal ray." (9 sentences in
# the test refs) -- original word not confidently recoverable.

# Doubled function words that are safe to collapse ("the the" -> "the").
_DEDUP = ("the", "a", "an", "in", "is", "of", "to", "and", "no", "for", "on")

_PH = "\x00P{}\x00"
_PH_RE = re.compile(r"\x00P(\d+)\x00")


def _protect(text):
    """Replace x-ray variants and the lowercase 'ray of' idiom with
    placeholders; return (protected_text, restore_fn)."""
    saved = []

    def save(m):
        saved.append(m.group(0))
        return _PH.format(len(saved) - 1)

    t = re.sub(r"x[- ]ray", save, text, flags=re.IGNORECASE)
    t = re.sub(r"\bray of\b", save, t)

    def restore(s):
        return _PH_RE.sub(lambda m: saved[int(m.group(1))], s)

    return t, restore


def _repair(t):
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r" +([,.;:!?])", r"\1", t)
    for w in _DEDUP:
        t = re.sub(r"\b%s %s\b" % (w, w), w, t, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", t).strip()


def clean_ray(text, aggressive=False):
    """Clean one report. Returns (cleaned_text, changes) where changes is a
    list of (rule_name, count) for auditing. Default mode applies RAY_RULES
    only; aggressive also removes any remaining standalone lowercase 'ray'."""
    if not text:
        return text, []
    work = text
    changes = []
    for pat, rep in RAY_RULES:
        work, n = re.subn(pat, rep, work, flags=re.IGNORECASE)
        if n:
            changes.append((pat, n))
    if aggressive:
        work, restore = _protect(work)
        work, n = re.subn(r"\bray\b", "", work)  # lowercase 'ray' only
        if n:
            changes.append(("generic \\bray\\b (aggressive)", n))
        work = restore(work)
    return _repair(work), changes


def ray_contexts(text, window=3):
    """Return [(context_key, case)] for every standalone 'ray' token in text.
    x-ray never counts (it tokenizes as one hyphenated token)."""
    out = []
    if not text:
        return out
    toks = re.findall(r"[\w'-]+", text)
    low = [t.lower() for t in toks]
    for i, t in enumerate(low):
        if t != "ray":
            continue
        pre = " ".join(low[max(0, i - window):i])
        post = " ".join(low[i + 1:i + 1 + window])
        out.append(((pre + " [RAY] " + post).strip(), t))
    return out


def _gpt_value(rec):
    for c in rec["conversations"]:
        if c.get("from") == "gpt" or "gpt" in c:
            return c["value"]
    raise SystemExit("record has no gpt turn: %s" % rec.get("id"))


def report(sources, top=25):
    """sources: list of (name, [texts]). Prints per-source totals + top
    contexts + how many places the current rules would change."""
    ctx = Counter()
    per_source = {}
    for name, texts in sources:
        n = 0
        for t in texts:
            for key, case in ray_contexts(t):
                ctx[key] += 1
                n += 1
        per_source[name] = n
    changed = 0
    for name, texts in sources:
        for t in texts:
            _, ch = clean_ray(t)
            changed += sum(n for _, n in ch)
    print("ray occurrences: " + " | ".join("%s %d" % (k, v)
                                          for k, v in per_source.items()))
    print("current RAY_RULES would change: %d places" % changed)
    print("top %d contexts:" % top)
    for key, n in ctx.most_common(top):
        print("%4d  %s" % (n, key))
    return ctx


def report_sentences(sources, maxn=14, width=170):
    """Print the most frequent UNIQUE sentences containing a standalone
    'ray' (any case), so ambiguous contexts can be resolved from the full
    sentence. sources: list of (name, [texts])."""
    SENT = re.compile(r"(?<=[.!?])\s+|\n+")
    cnt = Counter()
    for _name, texts in sources:
        for t in texts:
            if not t:
                continue
            for s in SENT.split(t):
                s2 = re.sub(r"\s+", " ", s).strip()
                if re.search(r"\bray\b", s2, flags=re.IGNORECASE):
                    cnt[s2[:width]] += 1
    print("top %d unique sentences containing a standalone 'ray':" % maxn)
    for s, n in cnt.most_common(maxn):
        print("%3d  %s" % (n, s))
    return cnt


def selftest():
    c, ch = clean_ray("atherosclerotic calcifications of the aortic ray")
    assert c == "atherosclerotic calcifications of the aortic arch", c
    assert ch == [("aortic ray", 1)], ch

    c, ch = clean_ray("status post ray sternotomy ray with mild emphysema")
    assert c == "status post sternotomy with mild emphysema", c

    c, ch = clean_ray("no acute process on this x-ray")
    assert c == "no acute process on this x-ray" and ch == [], (c, ch)

    c, ch = clean_ray("no acute process on this X-RAY", aggressive=True)
    assert "X-RAY" in c and ch == [], (c, ch)

    c, ch = clean_ray("there is a ray ray opacities in the right lung")
    assert c == "there is a x-ray opacities in the right lung", c
    c, ch = clean_ray("no costophrenic ray blunting")
    assert c == "no costophrenic angle blunting", c
    c, ch = clean_ray("the ray examination consists of pa and lateral views")
    assert c == "the x-ray examination consists of pa and lateral views", c
    c, ch = clean_ray("there are no ray of a large pleural effusion")
    assert c == "there are no signs of a large pleural effusion", c
    c, ch = clean_ray("no ray of a pleural effusion", aggressive=True)
    assert c == "no signs of a pleural effusion", c

    c, ch = clean_ray("calcifications of the ray", aggressive=True)
    assert "ray" not in c and ch, (c, ch)

    c, ch = clean_ray("the the heart is enlarged , and mild emphysema")
    assert c == "the heart is enlarged, and mild emphysema", c

    c, ch = clean_ray("no aortic ray")
    assert c == "no aortic arch", c

    assert ray_contexts("no ray of a pleural effusion") == [
        ("no [RAY] of a pleural", "ray")]
    assert ray_contexts("this x-ray is normal") == []
    print("rayclean selftest: all OK")


def main():
    ap = argparse.ArgumentParser(description="ray artifact cleanup (auditable)")
    ap.add_argument("--report", action="store_true",
                    help="scan test refs + baseline + finetuned pred files")
    ap.add_argument("--sentences", action="store_true",
                    help="most frequent FULL sentences containing 'ray' "
                         "(resolves ambiguous contexts)")
    ap.add_argument("--sentences-n", type=int, default=14)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--text", help="clean a single string")
    ap.add_argument("--aggressive", action="store_true",
                    help="also remove remaining standalone lowercase 'ray'")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if args.text is not None:
        c, ch = clean_ray(args.text, aggressive=args.aggressive)
        print(c)
        print("(changes: %s)" % (ch or "none"))
        return
    if args.report or args.sentences:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import eval_llavarad as ev
        with open(ev.TEST_JSON) as f:
            test = json.load(f)
        refs = [_gpt_value(r) for r in test]
        base = ev.load_jsonl(ev.BASE_FILE)
        ft = ev.load_jsonl(ev.FT_FILE)
        sources = [("test refs", refs),
                   ("baseline preds", [r["prediction"] for r in base]),
                   ("finetuned preds", [r["prediction"] for r in ft])]
        if args.report:
            report(sources, top=args.top)
        if args.sentences:
            report_sentences(sources, maxn=args.sentences_n)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
