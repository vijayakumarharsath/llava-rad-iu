# Fine-tuning LLaVA-Rad for Chest X-ray Report Drafting

A solo M.Sc. (AI & Robotics) portfolio project.
**The model produces DRAFT reports for human review. It is never a diagnosis.**

---

## 1. What this project does

Given a chest X-ray and a short clinical indication, the model drafts a short
radiology report. I took the publicly available LLaVA-Rad model (Microsoft,
vision-language model for radiology) and fine-tuned it on the open IU X-ray
collection so it learns that dataset's reporting style. I then evaluated it
against the official, un-fine-tuned model on a held-out test set, and reviewed
the errors by hand case by case.

The goal was not to build a clinical tool. The goal was to fine-tune a real
open vision-language model end to end on a single 16 GB GPU, measure it
honestly, and understand exactly where it gets things right and wrong.

## 2. Data

- Source: the open **IU X-ray (NLMCXR)** collection, 75.2 GB DICOM archive.
- Preprocessing: DICOM converted to 518x518 grayscale images, each paired with
  its radiology report text.
- Split: **per patient** (no patient appears in both sets, so no image
  overlap): **2,837 pairs for training, 500 pairs for testing**.
- Input format: "Provide a description of the findings in the radiology image
  [given the following indication: ...]"
- Known quirk: ages and numbers are anonymized as the word "ray" in the source
  dataset. That placeholder occasionally leaks into model outputs (see the
  error review, cases CXR2 and CXR3416).

## 3. Model and fine-tuning

Base model: **LLaVA-Rad** = Vicuna-7B LLM + BiomedCLIP-CXR (518) vision
encoder + an image-to-text projector (768 -> 4096, 2-layer MLP).

What was trained vs frozen:

- Trained: LoRA adapters on all attention and MLP projections
  (q, k, v, o, gate, up, down) plus the image-to-text projector
- Frozen: the BiomedCLIP-CXR vision encoder and the base LLM weights

Recipe (single RTX 5060 Ti, 16 GB):

- QLoRA: 4-bit NF4 quantization with double quantization, fp16 compute
- LoRA: r=64, alpha=128, dropout 0.05
- Optimizer: AdamW, learning rate 1e-4, cosine schedule, 3% warmup, no weight decay
- Batch: effective 16 (1 image per step x 16 gradient accumulation)
- **3 epochs = 531 iterations** (177 per epoch: 2,837 / 16)
- bf16 mixed precision, TF32, gradient checkpointing, max sequence length 2048
- Speed: about 51 s per iteration; the final logged segment (steps 300 -> 531)
  took 11,800 s (about 3 h 17 m); final training loss 0.212 at epoch 3.0

Inference: 4-bit base + unmerged LoRA adapter + fp16 projector. The LoRA
adapter is deliberately NOT merged: merging would upcast every LoRA layer to
16-bit and push the model past the 16 GB budget. Cost: about 5 GB VRAM on one
GPU, about 3 s per image.

Why QLoRA: it is what makes a 7B LLM plus a frozen vision encoder trainable on
a single consumer 16 GB card.

## 4. Evaluation setup

- **500 held-out test X-rays** (the test patients were never in training).
- Generation: copied verbatim from the official LLaVA-Rad evaluation script:
  greedy decoding (no sampling), max 256 new tokens, conversation template v1,
  images in float16 on the GPU.
- Metrics:
  - **RadGraph** (radgraph-xl, reward level "all"): three components
    - rg_e: entity match
    - rg_er: entity + relation-exists match (the headline metric)
    - rg_bar_er: entity + full relation match
  - **ROUGE-L F1** (word level)
- Why rg_er is the headline: before trusting the pipeline I reproduced the
  published baseline number. The official model's reported score was 0.287;
  my baseline run on the same 500 X-rays gave **0.2877**, so the scoring path
  is validated.

## 5. Results (n = 500, fine-tuned vs official baseline)

| Metric        | Baseline | Fine-tuned | Delta    | Relative |
|---------------|----------|------------|----------|----------|
| RadGraph rg_e     | 0.3077 | 0.3526 | +0.0449 | +14.6% |
| RadGraph rg_er (headline) | 0.2877 | 0.3342 | +0.0465 | +16.2% |
| RadGraph rg_bar_er | 0.2225 | 0.2797 | +0.0572 | +25.7% |
| ROUGE-L F1        | 0.2037 | 0.2673 | +0.0636 | +31.2% |

All four metrics improve. But aggregate numbers can hide what the model
actually does, so every conclusion below is backed by a manual case review.

## 6. Per-sample review (manual)

Method: from the 500 test cases I pulled 16 for close reading: the 5 biggest
gains, the 5 biggest drops, 4 where both models scored low, and 2 random. I
read all three texts of each case (reference, baseline, fine-tuned) and
judged which report I would actually trust.

Verdicts: **fine-tuned better 6, baseline better 2, tie 6, both poor 2.**

### What the fine-tuned model got better at

**Normal studies, house style.** On normal X-rays the fine-tuned model
reproduces the dataset's canonical report almost verbatim, e.g. CXR3330
(score 0.0 -> 1.0, word-for-word match with the reference) and CXR2516, where
the fine-tuned output matches every sentence of the reference except the
scarring sentence.

### What still goes wrong

**1. The pathology can still be missed even in a "win" (CXR2516).**
The reference says "stable mild hyperinflation, right apical
pleural-parenchymal irregularities compatible with scarring". Both models
miss that sentence. The fine-tuned model wins (0.06 -> 0.80) by matching the
rest of the report, not by seeing the scarring. A high score does not mean the
abnormality was found.

**2. The metric can be wrong about which model is better (CXR975).**
The reference describes emphysematous changes with hyperinflation. The
fine-tuned model says "Emphysematous changes are noted. The lungs are mildly
hyperexpanded." The baseline says "The lungs appear clear." Clinically the
fine-tuned model is right and the baseline missed the finding. The metric
scores it the other way: fine-tuned 0.000, baseline 0.38. RadGraph is
token-strict and rewards phrase coverage, so some score differences are
phrasing, not meaning.

**3. Per-finding review of a hard case (CXR3312).**
Reference: right middle lobe infiltrate, left lung clear, normal heart,
tortuous aorta.

| Finding    | Reference                | Fine-tuned                  |
|------------|--------------------------|-----------------------------|
| Right lung | patchy infiltrate, RML   | RML airspace disease (right) |
| Left lung  | clear                    | not mentioned               |
| Heart      | normal size              | "enlarged" (wrong)          |
| Aorta      | tortuous                 | "atherosclerotic calcifications" (different finding) |

Both models scored 0.000 by RadGraph. The fine-tuned model gets the main
finding (the lobe) right; the baseline names the wrong lobe (right lower). But
the fine-tuned model also hallucinates an enlarged heart. Each model wins
something and loses something: that is exactly why the output is a draft for a
human, not a diagnosis.

**4. Honest misses: defaulting to normal (CXR3416, CXR3521, CXR3479).**
In CXR3416 the reference shows left basilar atelectasis and a calcified nodule;
the baseline catches the nodule, the fine-tuned model says "the lungs are
clear" and adds aortic calcifications that are not in the reference. In
CXR3521 the reference shows biapical bullous emphysema and both models say the
lungs are clear. The fine-tuned model has a mild "default to normal" bias on
abnormal studies: when it is unsure, it falls back to the clean template.

**5. Anonymization leakage (CXR2, CXR3416).**
The dataset replaces numbers with "ray". The model learned the placeholder and
sometimes emits it: "ray sternotomy ray are again noted". It knows the
patient had a sternotomy, but the report reads broken.

### Overall pattern

- Gains concentrate on normal studies (the model learned the reporting style).
- On abnormal findings the two models are comparable; each wins different cases.
- The fine-tuned model's main failure mode is defaulting to normal.
- Metric score and clinical quality are related but not the same thing.

## 7. Live demo on a never-seen image

The terminal tool (see below) can take any image file. I ran it on a real
chest X-ray DICOM from outside the dataset (a private file provided for a
mentor demo; no data left the machine). The model produced its house-style
normal report in about 3.5 s. This is a pipeline demo on fully out-of-
distribution data: the model only trained on the open IU dataset (US
population), so such outputs must be treated as unvalidated drafts.

## 8. Limitations

- Single dataset, no external validation yet.
- RadGraph is token-strict: some score moves are phrasing, not semantics.
- 4-bit + unmerged LoRA is a 16 GB VRAM compromise, not a final research setup.
- Greedy generation with a 256 token cap.
- Default-to-normal bias on abnormal studies (the main real-world risk).
- "ray" anonymization placeholder leaks into some outputs.
- 500 test cases is a reasonable but modest held-out set.
- This is a draft-report generator for human review. It is not a diagnosis.

## 9. Reproducibility

Everything ran on one machine (Ubuntu 24.04, 2x RTX 5060 Ti 16 GB, the
training and eval used one GPU).

- Model artifacts: `~/llava-rad-checkpoints/llavarad-iu/`
  (LoRA adapter, fine-tuned projector, config). Base + official weights in
  `~/llava-rad-checkpoints/`.
- Data: `~/iu-xray/llava_iu/` (train.json 2,837, test.json 500, images/).
- Evaluation script: `~/iu-xray/eval_llavarad.py`
  (modes: baseline / generate / score / all; wrote `metrics.json` and
  `review_shortlist.jsonl`).
- Results: `~/iu-xray/metrics.json`, predictions in
  `baseline_preds.jsonl` and `finetuned_preds.jsonl` (500 each).
- Terminal tool: `~/iu-xray/cxr_show.py` (browse the 500, step the 16 review
  cases, live predict on ids or on any png/dcm file, metrics table, training
  info). DICOM converter: `~/iu-xray/dcm2png.py`.
- Three small non-standard patches were needed for the old repo on a modern
  stack (deepspeed import guard in `llava/train/train.py`,
  `weights_only=False` on two legacy `torch.load` calls in the repo and the
  venv's trainer). They do not change model behavior.

## 10. Next steps

- Re-run the same pipeline on MIMIC-CXR (needs PhysioNet credentialing) as a
  larger, standard target dataset.
- External validation on a second collection.
- Optional: merge or distill the LoRA adapter for deployment if VRAM is no
  longer the constraint.
