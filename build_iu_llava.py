#!/usr/bin/env python3
"""
Step 3 — Build LLaVA-Rad-format image/report pairs from IU X-ray (Open-I).

Reads:
  ~/iu-xray/ecgen-radiology/*.xml    (reports with INDICATION/FINDINGS/IMPRESSION)
  ~/iu-xray/*.png                    (images; -1001 = frontal, -2001 = lateral)

Writes:
  ~/iu-xray/llava_iu/images/         (symlinks to the chosen images)
  ~/iu-xray/llava_iu/train.json      (LLaVA format, study-level split)
  ~/iu-xray/llava_iu/test.json

Decisions made here (tell me if you want any changed):
  - Frontal images only: prefer ids ending -1001; if a report has none,
    fall back to any image that isn't -2001.
  - Prompt uses LLaVA-Rad's own wording: "...given the following indication: X"
    or "...describe the findings in the radiology image." when no indication.
  - Target = FINDINGS section only (LLaVA-Rad generates findings, not impressions).
  - IU anonymization quirk fixed: "XXXX" -> "ray" (e.g. "chest x-XXXX" -> "chest x-ray").
  - Split by STUDY id (the CXR number) so all images of one study stay in one split.
    IU X-ray has no public patient ids — flagging this as a known dataset limitation.
"""

import glob
import json
import os
import random
import re
import sys
import xml.etree.ElementTree as ET

DATA = os.path.expanduser("~/iu-xray")
OUT = os.path.join(DATA, "llava_iu")
TEST_FRAC = 0.15
SEED = 42

PROMPT_TEMPLATE_IND = (
    "<image>\nProvide a description of the findings in the radiology image "
    "given the following indication: {indication}"
)
PROMPT_TEMPLATE_PLAIN = (
    "<image>\nProvide a description of the findings in the radiology image."
)


def fix_iu_quirks(text):
    """XXXX is IU's anonymization of the word 'ray'."""
    if not text:
        return text
    return re.sub(r"\bX{2,}\b", "ray", text)


def section_text(root, label):
    """Returns the text inside an AbstractText with the given Label."""
    for el in root.iter():
        if el.tag == "AbstractText" and (el.get("Label") or "").upper() == label:
            return fix_iu_quirks(" ".join(el.text.split()) if el.text else "")
    return ""


def main():
    xmls = sorted(glob.glob(os.path.join(DATA, "ecgen-radiology", "*.xml")))
    print(f"found {len(xmls)} report files")

    # ---- parse reports ----
    reports = []  # dicts: study, indication, findings, impression, image_ids
    for path in xmls:
        try:
            root = ET.parse(path).getroot()
        except Exception:
            continue
        findings = section_text(root, "FINDINGS")
        if not findings:
            continue
        indication = section_text(root, "INDICATION")
        impression = section_text(root, "IMPRESSION")
        ids = [el.get("id") for el in root.iter("parentImage") if el.get("id")]
        study = re.match(r"(CXR\d+)", os.path.basename(path)) or None
        # study id from first image id if filename doesn't start with CXR
        if study is None and ids:
            study = re.match(r"(CXR\d+)", ids[0])
        reports.append({
            "study": study.group(1) if study else f"unk_{len(reports)}",
            "indication": indication,
            "findings": findings,
            "impression": impression,
            "ids": ids,
        })
    print(f"reports with findings: {len(reports)}")

    # ---- match images (frontal preferred) ----
    pngs = {os.path.basename(p): p for p in glob.glob(os.path.join(DATA, "*.png"))}
    print(f"png files on disk: {len(pngs)}")

    def pick_image(ids):
        for suffix in ("-1001", "-1002", "-0001"):
            for i in ids:
                if i.endswith(suffix) and f"{i}.png" in pngs:
                    return f"{i}.png"
        for i in ids:
            if not i.endswith("-2001") and f"{i}.png" in pngs:
                return f"{i}.png"
        for i in ids:
            if f"{i}.png" in pngs:
                return f"{i}.png"
        return None

    pairs = []
    for r in reports:
        img = pick_image(r["ids"])
        if img is None:
            continue
        prompt = (
            PROMPT_TEMPLATE_IND.format(indication=r["indication"])
            if r["indication"]
            else PROMPT_TEMPLATE_PLAIN
        )
        pairs.append({
            "study": r["study"],
            "image": img,
            "prompt": prompt,
            "findings": r["findings"],
        })
    print(f"image/report pairs built: {len(pairs)}")

    # ---- split by study ----
    studies = sorted({p["study"] for p in pairs})
    random.Random(SEED).shuffle(studies)
    n_test = max(1, int(len(studies) * TEST_FRAC))
    test_studies = set(studies[:n_test])
    print(f"studies: {len(studies)} -> train {len(studies)-n_test}, test {n_test}")

    def to_llava(p):
        return {
            "id": p["image"].replace(".png", ""),
            "image": p["image"],
            "conversations": [
                {"from": "human", "value": p["prompt"]},
                {"from": "gpt", "value": p["findings"]},
            ],
        }

    train = [to_llava(p) for p in pairs if p["study"] not in test_studies]
    test = [to_llava(p) for p in pairs if p["study"] in test_studies]

    # ---- write output with symlinked images ----
    os.makedirs(os.path.join(OUT, "images"), exist_ok=True)
    for split_name, rows in (("train", train), ("test", test)):
        with open(os.path.join(OUT, f"{split_name}.json"), "w") as f:
            json.dump(rows, f, indent=1)
        print(f"{split_name}.json: {len(rows)} rows")

    used = {r["image"] for r in train + test}
    made = 0
    for img in used:
        link = os.path.join(OUT, "images", img)
        if not os.path.exists(link):
            try:
                os.symlink(pngs[img], link)
                made += 1
            except OSError:
                pass
    print(f"symlinked {made} images into {OUT}/images")

    print("\nDone. Use with:")
    print(f"  --image_folder {OUT}/images")
    print(f"  --query_file  {OUT}/test.json   (or train.json)")
    print("\nSanity preview (first test row):")
    print(json.dumps(test[0], indent=1)[:400])


if __name__ == "__main__":
    main()
