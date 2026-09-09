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
- Known quirk: the source reports are systematically corrupted - clinical
  terms are replaced with the word "ray" (a word-level substitution bug in
  the corpus generation; investigation in Section 10.1). The v1 model learned
  this and emitted 194 "ray" occurrences in its test outputs; the v2 round
  addressed it (Section 10).

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
- "ray" anonymization placeholder leaks into some outputs (investigated and
  partially fixed in the v2 round, Section 10).
- Finding labels come from a deterministic rule-based labeler (the official
  CheXbert weights are no longer public); it approximates CheXbert, not
  equals it.
- Entity FP ("ungrounded") measures grounding in the reference, not false
  findings per se - the reference can omit true findings.
- The "ray" cleanup leaves a documented residual where the original word is
  not recoverable; one uncovered pattern ("spine there is ray deformity")
  survives in v2 outputs and drove a repetition-loop failure (CXR3416).
- v2 was evaluated, not adopted: its results are a measured trade-off and v1
  stays the headline model.
- v3 (full bf16) is best on the aggregate axes; the 3-epoch
  checkpoint's "ray" count is slightly below v1's (186 vs 194),
  the 6-epoch one's is up (317 vs 80 for ray-cleaned v2)
  (Section 11.5).
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
- v2 round: `train_v2.json` + `v2_weights.json` (cleaned data, oversampling
  weights), predictions `finetuned_preds_v2.jsonl`, results
  `metrics_v1.json` / `metrics_v2.json`,
  `entity_metrics_{v1,v2,v1clean,v2clean}.json`,
  `findings_metrics_{v1,v2,v1clean,v2clean}.json`. Scripts (all in the repo):
  `rayclean.py`, `findlabel.py`, `eval_extra.py`, `prep_v2.py`,
  `patch_train_sampler.py`, `run_train_v2.sh`.
- Terminal tool: `~/iu-xray/cxr_show.py` (browse the 500, step the 16 review
  cases, live predict on ids or on any png/dcm file, metrics table, training
  info). DICOM converter: `~/iu-xray/dcm2png.py`.
- Three small non-standard patches were needed for the old repo on a modern
  stack (deepspeed import guard in `llava/train/train.py`,
  `weights_only=False` on two legacy `torch.load` calls in the repo and the
  venv's trainer). They do not change model behavior.

## 10. Follow-up round: the hallucination feedback (08-28/29)

After v1, my mentor flagged that the fine-tuned model "is hallucinating" and
asked for evaluation beyond text overlap. This section is that follow-up as a
full second round: find the cause of the complaint, build metrics that can
measure it, fix the data, retrain, and compare on every axis.

### 10.1 Root cause: the training data is corrupted

The source reports contain a systematic corruption: clinical terms are
replaced with the literal word "ray" ("The ray are intact." = ribs,
"calcified mediastinal lymph ray" = lymph nodes, "there are no ray of a large
pleural effusion" = no signs). Evidence that this is a word-level
substitution bug in the corpus generation, not de-identification:

- De-identification removes names, dates, and IDs with bracketed
  placeholders. It does not replace the word "ribs".
- Identical corrupted sentences appear verbatim in dozens of reports, so the
  bug is baked into the source, not per patient.
- "The heart and lungs have ray ray in the interval." = "have no change in
  the interval": the tool replaced words individually ("no" -> "ray",
  "change" -> "ray").
- "ray" is not a standard anonymization token; the exact origin is unknown
  (worth one email to the data provider).

The v1 model faithfully learned the corrupted style: 194 "ray" occurrences in
its 500 test outputs (official baseline: 0). A radiologist reading "the ray
are intact" sees gibberish and files it under hallucination, even though the
model was reproducing its training data. That is the measurable part of the
complaint.

**Cleanup (`rayclean.py`).** 12 auditable context rules recover the original
word where the context determines it ("ray are" -> "ribs are", "aortic ray"
-> "aortic arch", "no ray of" -> "no signs of", ...). Where the context does
not determine the word, the rule set refuses to guess: "Normal ray." (9
reference sentences) and the "spine there is ray deformity" pattern (almost
certainly "no deformity") are left untouched and documented. The rules cover
140 of the 334 "ray" occurrences in the test references; 730 of the 2,837
training pairs changed. Every rule is listed in the script; the change set is
auditable.

### 10.2 New metrics: false and missed findings

RadGraph and ROUGE measure overlap with the reference text, not clinical
correctness. Two new axes (`eval_extra.py`; it imports the frozen v1 eval
script, which was not modified):

1. **Entity-level FP/FN from RadGraph annotations.** Per report pair:
   entities in the prediction but not in the reference (ungrounded) and
   entities in the reference but not in the prediction (missed). Entity
   identity is the same (tokens, label) key RadGraph uses for rg_e, so this
   axis is consistent with the overlap metrics. Split overall / abnormal
   references / normal references. An ungrounded entity is not necessarily a
   false finding (the reference can omit a true finding), so this axis
   measures grounding, not diagnosis.
2. **Finding-level F1 on the 14 CheXbert conditions.** The official CheXbert
   classifier weights are no longer publicly available (the HF repo is gone
   as of 2026-08-28), so a deterministic rule-based labeler (`findlabel.py`)
   implements the same 14-label set with negation windows; it is documented
   as an approximation of CheXbert, not CheXbert itself. Per-label
   TP/FP/FN/F1 for baseline and fine-tuned, macro F1, plus false findings
   and missed findings on abnormal references and false alarms on normal
   references.

### 10.3 v2 design

Same 2,837 pairs, same ids, same patient split, same recipe (QLoRA r=64, 3
epochs, 531 iterations, ~51 s/iteration). Two changes only:

1. Training reports ray-cleaned (730 pairs changed; `train.json` untouched,
   cleaned copy in `train_v2.json`).
2. Abnormal reports (73.0% of the data per the rule labeler) drawn 2x
   (effective 84.4%), via an env-gated WeightedRandomSampler in the trainer.
   The sampler draws exactly 2,837 samples per epoch, so the iteration count
   is identical to v1.

Run: 3 epochs in 27,148 s (7 h 32 m); final epoch-3 loss ~0.38. (Not
comparable to v1's final-segment 0.212: the data is different, and 84%
abnormal is harder material than templated normals.)

### 10.4 Results (n = 500)

Pipeline re-verified before trusting any number: the official baseline
reproduced exactly (rg_e/rg_er/rg_bar_er/ROUGE-L = 0.3077/0.2877/0.2225/
0.2037) and a fresh scoring run reproduced v1's official numbers exactly.

vs the original (corrupted) references:

| Axis | Baseline | v1 | v2 |
|---|---|---|---|
| RadGraph rg_e | 0.3077 | 0.3526 | 0.3411 |
| RadGraph rg_er | 0.2877 | 0.3342 | 0.3162 |
| RadGraph rg_bar_er | 0.2225 | 0.2797 | 0.2587 |
| ROUGE-L F1 | 0.2037 | 0.2673 | 0.2499 |
| Entity FP/report (abnormal refs) | 8.43 | 7.01 | 7.88 |
| Entity FN/report (abnormal refs) | 12.05 | 11.88 | 11.70 |
| CheXbert-14 macro F1 (rule labeler) | 0.2082 | 0.2331 | 0.2755 |
| Missed findings (abnormal refs) | 743 | 625 | 527 |
| False findings (abnormal refs) | 147 | 132 | 207 |
| "ray" occurrences in outputs | 0 | 194 | 80 |

vs ray-cleaned references (the fair comparison for the cleaned model):
v1 = 0.3536/0.3347/0.2810/0.2664, v2 = 0.3427/0.3177/0.2612/0.2504
(rg_e/rg_er/rg_bar_er/ROUGE-L). Cleaning the references moves these numbers
by ~0.001, so the v2-below-v1 overlap gap is real (v2's text moved off the
house style) and not an artifact of the corrupted references.

Per-label finding F1: v2 beats v1 on 10 of 14 labels (Pneumothorax 0.568 vs
0.521, Pleural Effusion 0.566 vs 0.458, Cardiomegaly 0.444 vs 0.278, Lung
Opacity 0.350 vs 0.218) and loses only Atelectasis (0.103 vs 0.188) and No
Finding (0.395 vs 0.474).

### 10.5 LLM judge on the 16-case review

The same 16 cases from the manual review, with reference + baseline + v1 +
v2 outputs, judged by a general-purpose LLM (unblinded: the model labels were
visible; disclosed here). Per case it picks the best report and gives a
one-line reason.

- Calibration against the signed-off human verdicts (v1 vs baseline):
  **16/16 agreement.**
- V2 vs v1 on the same 16: **2 better, 6 worse, 8 tie.**
  - Better: two real finding catches (CXR3479 diffuse-bilateral distribution,
    CXR1430 bibasilar opacities).
  - Worse: a false opacity on a normal study (CXR3209), wrong lobe +
    unsupported cardiomegaly (CXR3312), a missed emphysema (CXR975), a
    "Normal ray." artifact (CXR1974), a truncated report (CXR2516), and one
    degenerate case: CXR3416, where v2 catches the calcified granuloma that
    v1 missed but then repeats "There is ray deformity of the ... thoracic
    spine" about 15 times and cuts off at the token limit. That uncovered
    corruption pattern (the "spine there is ray deformity" family, ~13
    occurrences across v2 outputs) has gone into a repetition loop; it is the
    worst single output of either model.

### 10.6 Conclusion

v2 is an operating-point shift, not a clean win:

- What it buys: finding macro F1 +18% (0.2331 -> 0.2755), missed findings
  -16% (625 -> 527), 59% fewer "ray" artifacts (194 -> 80, all in the
  documented residual family), higher entity recall.
- What it costs: false finding labels +57% (132 -> 207), RadGraph/ROUGE down
  ~0.017-0.018 (real, not a reference artifact), 2-6-8 on the human-level
  16-case review, and a new repetition-loop failure mode.

**Decision: v1 remains the headline model; v2 is the documented experiment.**
The round answers the mentor's question with measurements: part of the
"hallucination" complaint was the model faithfully reproducing a corrupted
corpus (now quantified and reduced 194 -> 80, residual fully audited); the
rest is a genuine precision/recall trade, quantified on finding-level and
entity-level axes.

## 11. v3 round: full bf16, no quantization (09-07/08)

After the v1/v2 rounds, the plan was to stop quantizing: v3 trains and
runs in full bf16 everywhere (base LLM, LoRA, projector, and the frozen
vision encoder). The run also went to 6 epochs and used both GPUs, so
this round changes three things at once; the run's own epoch-3
checkpoint is evaluated separately to keep the comparisons honest
(Section 11.4).

### 11.1 Setup

- Same base model, same 2,837 training pairs, same patient split, same
  frozen 500-case test set, same frozen eval pipeline (the v1 eval
  script was not modified; Section 4).
- Full bf16: no 4-bit quantization anywhere - base LLM, LoRA,
  projector, vision tower.
- LoRA r=64, alpha=128, dropout 0.05; projector fine-tuned;
  BiomedCLIP-CXR 518 frozen.
- AdamW, lr 1e-4, cosine with 3% warmup, weight decay 0, grad clip 1.0.
- Effective batch 16, held constant across all rounds: v1/v2 used
  1 x 16 gradient accumulation on one GPU; v3 uses 1 x 8 accumulation
  x 2 GPUs under DeepSpeed ZeRO stage 3 with CPU optimizer offload.
- 2x RTX 5060 Ti 16 GB, gradient checkpointing, max sequence length
  2048.
- 6 epochs = 1,062 iterations in ~16.5 h; final training loss 0.3998
  (whole-run mean; v2's last epoch was ~0.38 on its harder cleaned
  data, and v1's reported figure was a late-segment value, so
  cross-round loss comparison is indicative at best).

Two operational incidents, no recipe impact: an overnight run died when
the SSH session dropped and the launcher got SIGHUP (nohup does not
protect a launcher that execs into torchrun's elastic agent); it was
relaunched with `setsid` and finished unattended. A second eval launched
while the first still held the GPUs hit an out-of-memory error; the bf16
evals now run strictly one at a time.

### 11.2 An eval dtype bug, found and fixed

The first v3 eval crashed on sample 1: during training the image tensors
reached the bf16 projector as bf16 (the bf16 DeepSpeed engine boundary),
but the eval path fed fp16 - the frozen eval script hardcodes
`image_tensor.half()`. The bf16 projector hit a dtype mismatch. The fix
matches the eval to the training dtypes (bf16 vision tower, bf16
projector, image tensor cast to bf16 before `generate`) via a small
wrapper; the frozen eval script itself was not touched. Lesson recorded:
eval dtypes must match training dtypes.

### 11.3 Results (n = 500)

"v3 6ep" is the run's final checkpoint; "v3 3ep" is the same run's
epoch-3 checkpoint (its projector was extracted from the DeepSpeed
state before it could be loaded). Same table as Section 10.4 with the
two v3 columns added:

| Axis | Baseline | v1 | v2 | v3 3ep | v3 6ep |
|---|---|---|---|---|---|
| RadGraph rg_e | 0.3077 | 0.3526 | 0.3411 | 0.3783 | 0.3647 |
| RadGraph rg_er | 0.2877 | 0.3342 | 0.3162 | 0.3503 | 0.3366 |
| RadGraph rg_bar_er | 0.2225 | 0.2797 | 0.2587 | 0.2904 | 0.2880 |
| ROUGE-L F1 | 0.2037 | 0.2673 | 0.2499 | 0.2794 | 0.2781 |
| Entity FP/report (abnormal refs) | 8.43 | 7.01 | 7.88 | 7.71 | 8.72 |
| Entity FN/report (abnormal refs) | 12.05 | 11.88 | 11.70 | 11.20 | 11.29 |
| Entity precision (abnormal refs) | 0.3477 | 0.3994 | 0.3806 | 0.4091 | 0.3756 |
| Entity recall (abnormal refs) | 0.2717 | 0.2817 | 0.2928 | 0.3227 | 0.3171 |
| CheXbert-14 macro F1 (rule labeler) | 0.2082 | 0.2331 | 0.2755 | 0.2723 | 0.2726 |
| Missed findings (abnormal refs) | 743 | 625 | 527 | 582 | 573 |
| False findings (abnormal refs) | 147 | 132 | 207 | 181 | 218 |
| False alarms (normal refs) | 51 | 40 | 60 | 37 | 53 |
| "ray" tokens in model outputs | 0 | 194 | 80 | 186 | 317 |

### 11.4 Attribution: what the precision bought, what the extra epochs did

- Precision, at equal epochs: bf16 at 3 epochs beats both 4-bit 3-epoch
  runs on rg_er (0.3503 vs 0.3342 for v1, 0.3162 for v2). The spread
  between two runs of the same v1/v2 recipe is 0.018, so the bf16 gain
  is real and comparable in size to run-to-run variance.
- Epochs, at equal precision: within the same bf16 run, the 6-epoch
  checkpoint is below the 3-epoch one on all four overlap axes
  (rg_er 0.3366 vs 0.3503), and it emits more ungrounded entities
  (8.72 vs 7.71 per abnormal report - worse than the un-fine-tuned
  baseline's 8.43) and more false findings (218 vs 181). The extra three
  epochs look like mild overfitting to the 2,837 pairs, not a gain.

Honest caveats on these comparisons: the 3ep-vs-6ep gap compares two
checkpoints of one run selected on the same 500-case set that is
reported on, so part of it may be selection. The effective batch is
identical across all rounds (16), so no batch-size caveat is needed.

### 11.5 The "ray" artifact: corrected count, and what it actually says

An early version of these results quoted 1000/1130 "ray" tokens for the
two v3 checkpoints. That number came from a whole-line grep over the
prediction files, and those files store one line per case as
{id, query, reference, prediction}: the grep also counted the stored
query and reference text. The test corpus itself carries the "ray"
corruption in both (Section 10.1: indications like "ray-year-old",
references like "no ray of a large pleural effusion"), so a whole-line
grep lands at 1000-1130 for the v3 files and 1011 for the v1 file -
the same order of magnitude for every model, which is the fingerprint
of counting the stored corpus, not the model output. Quoting it as a
bf16 regression was a measurement error; it is corrected here.

The Section 11.3 table now shows the model output only, counted
identically for every model (standalone "ray" tokens, case-insensitive,
hyphenated x-ray never counted - the same rule used for v1/v2):
v1 194, v2 80, v3 3ep 186, v3 6ep 317. For the record, under the same
token rule the stored text in each file contributes 272 query tokens
and 334 reference tokens; the raw grep additionally counts "ray"
inside hyphenated compounds (ray-year-old, ray-ray) and is
case-sensitive, which is why it reads higher still.

The corrected story: the 3-epoch bf16 model shows no ray regression -
186 is slightly below v1's 194 and about 2.3x the ray-cleaned v2's 80.
The 6-epoch checkpoint does rise (317, 1.6x v1, ~4x v2), which fits the
mild-overfitting reading of Section 11.4 rather than any precision
effect. Corpus cleanup remains the proven mitigation (v1 194 -> v2 80).
The ray-cleaned re-scores of both v3 checkpoints were computed
alongside the published ones (the `*clean` result files); the cleaned
comparison goes into the repo with the rest (Section 12).

### 11.6 Conclusion

The 3-epoch bf16 checkpoint is the best model of the three rounds on
every overlap axis (rg_er 0.3503, +0.0626 over the reproduced baseline),
on entity precision and recall, and on false alarms. It is the model
taken forward to external validation. Its known cost vs v1 is a higher false-finding count (181 vs 132); its
"ray" token count is, if anything, slightly lower (186 vs 194;
Section 11.5).

## 12. Next steps

- Validate the 3-epoch bf16 checkpoint (v3 3ep, the best model so far)
  on MIMIC-CXR (needs PhysioNet credentialing) as a larger, standard
  target dataset.
- Tabulate the ray-cleaned re-scores of both v3 checkpoints and add the
  v3 result files to the repo.
- Email the data provider about the source "ray" corruption (upstream fix for
  any future retrain).
- External validation on a second collection.
- Optional: merge or distill the LoRA adapter for deployment if VRAM is no
  longer the constraint.
