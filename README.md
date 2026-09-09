# LLaVA-Rad fine-tuned on open-IU chest X-rays

Fine-tune of [microsoft/LLaVA-Rad](https://github.com/microsoft/LLaVA-Rad)
(vicuna-7b-v1.5 + frozen BiomedCLIP-CXR 518 vision tower) on open-IU chest
X-ray image-report pairs, evaluated against the official un-fine-tuned
baseline on a held-out, per-patient test set (n = 500).

This produces draft reports for human review. It is research software and is
never a diagnosis.

## Runs

All runs are LoRA (r=64, alpha=128, dropout 0.05), lr 1e-4 cosine with 3%
warmup, effective batch 16, max length 2048, on 2x RTX 5060 Ti 16 GB:

- v1, v2: 4-bit QLoRA, 1x16 micro-batch, 1 GPU
- v3: full bf16 LoRA, 1x8x2 micro-batch, DeepSpeed ZeRO-3 + CPU offload

Best run: **v3, 3 epochs (checkpoint-531)**. The 6-epoch v3 run is kept as an
overfitting reference: it scores worse on every axis and emits 1.7x more
"ray" artifact tokens.

## Results (n = 500 held-out test X-rays, same scoring pipeline for all)

| model | rg_e | rg_er | rg_bar_er | rouge_l |
|---|---|---|---|---|
| baseline (official LLaVA-Rad) | 0.3077 | 0.2877 | 0.2225 | 0.2037 |
| v1 (QLoRA, 3ep) | 0.3526 | 0.3342 | 0.2797 | 0.2673 |
| v2 (QLoRA, 3ep) | 0.3411 | 0.3162 | 0.2587 | 0.2499 |
| **v3 3ep (bf16, checkpoint-531)** | **0.3783** | **0.3503** | **0.2904** | **0.2794** |
| v3 6ep (bf16, final) | 0.3647 | 0.3366 | 0.2880 | 0.2781 |

Headline metric: rg_er (RadGraph entity-relationship partial reward). The
local baseline run reproduces the published 0.287, which validates the
scoring path. Entity-level (abnormal reports): FP/report 8.429 -> 7.710,
FN/report 12.046 -> 11.201, precision 0.3477 -> 0.4091, recall
0.2717 -> 0.3227 (baseline -> v3 3ep). Full tables, a clean-variant
re-scoring, and a 16-case manual review are in `writeup.md`.

Known data quirk: ages and numbers are anonymized as the literal word "ray"
in the source reports, so fine-tuned outputs occasionally emit it
(counted on the prediction field only, standalone tokens, hyphenated
"x-ray" excluded: baseline 0, v1 194, v2 80, v3 3ep 186, v3 6ep 317).
`rayclean.py` removes the artifacts; re-scoring cleaned outputs changes
rg_er by <= 0.001 and no conclusion flips.

## Weights

Adapter weights for the v3 checkpoint-531 model (LoRA adapter +
`non_lora_trainables.bin` projector) are on Hugging Face:
`vijayakumarharsath/llava-rad-iu-v3`. Load with the bf16 loaders in this
repo (`run_v3_eval.py` for IU, `mimic_eval.py` for MIMIC) - both do the full
load: vicuna-7b-v1.5 base + LLaVA config patch + projector install + unmerged
adapter, all in bf16 to match training.

## Repository layout

    writeup.md            full write-up: data, training, evaluation,
                          manual review, limitations, reproducibility
    show_results.py       prints the results tables from the metrics files
    eval_llavarad.py      frozen eval script (load, generate, RadGraph/ROUGE
                          scoring, metrics.json); the committed results were
                          produced with this exact version
    eval_extra.py         extra scoring: findings labels, RadGraph entities
    run_v3_eval.py        bf16 model loader + IU eval for the v3 checkpoints
    rayclean.py           removes "ray" scrubber artifacts from reports
    findlabel.py          rule-based CheXbert-style findings labeler
    metrics_v3/           entity + findings metrics for the v3 checkpoints
    mimic_subset.py       builds the official MIMIC-CXR test subset (500
                          samples) from the PhysioNet annotation json
    mimic_eval.py         MIMIC-CXR eval for our bf16 checkpoint
    mimic_score.py        MIMIC scoring, same pipeline as the IU eval
    ds_config_v3.json     DeepSpeed ZeRO-3 config used for v3
    requirements.txt      dependencies (install torch cu128 first)

External validation: the `mimic_*.py` scripts run the official 500-sample
MIMIC-CXR test split for both the official baseline and our checkpoint-531
model. They require the MIMIC-CXR files from PhysioNet (DUA + offline
token); the data is not included in this repo.

## Setup

    python3.12 -m venv ~/llavarad-env
    ~/llavarad-env/bin/pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
    ~/llavarad-env/bin/pip install -r requirements.txt

Weights, all from Hugging Face:

    huggingface-cli download lmsys/vicuna-7b-v1.5 --local-dir ~/llava-rad-checkpoints/vicuna-7b-v1.5
    huggingface-cli download microsoft/llava-rad --local-dir ~/llava-rad-checkpoints/llavarad
    huggingface-cli download vijayakumarharsath/llava-rad-iu-v3 --local-dir ~/llava-rad-checkpoints/llava-rad-iu-v3/checkpoint-531

The evaluation scripts take the checkpoint directory, base model, vision
tower config/weights and prediction files as arguments; see each script's
`--help` and `writeup.md` (reproducibility section) for the exact commands.
Retraining is described in `writeup.md` and uses the official
microsoft/LLaVA-Rad repo plus `ds_config_v3.json`.

## What is NOT in this repo (and why)

- model weights (multi-GB) -> Hugging Face, above
- raw image archives and any patient-identifiable files -> never committed,
  see `.gitignore`
- MIMIC-CXR data -> PhysioNet, behind a data-use agreement
- the local web UI and DICOM browsing tools -> tied to the local machine
  layout, not part of the research pipeline

## Data notes

- source: open IU chest X-ray collection (NLMCXR). 3955 reports with
  findings; 3337 pair with a usable frontal image (2837 train / 500 test),
  split per patient, no patient appears in both sets
- the "ray" anonymization artifact is documented above and in `writeup.md`
