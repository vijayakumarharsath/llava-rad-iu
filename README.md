# LLaVA-Rad (QLoRA) on IU chest X-rays

Fine-tune of [microsoft/llava-rad](https://huggingface.co/microsoft/llava-rad)
on the open IU chest X-ray collection, trained with QLoRA on a single 16 GB
GPU, and evaluated against the official un-fine-tuned baseline on a held-out,
per-patient test set.

This produces draft reports for human review. It is research software and is
never a diagnosis.

## Results (n = 500 held-out test X-rays)

| model | rg_e | rg_er | rg_bar_er | rouge_l |
|---|---|---|---|---|
| baseline (official LLaVA-Rad) | 0.3077 | 0.2877 | 0.2225 | 0.2037 |
| fine-tuned (QLoRA, this repo) | 0.3526 | 0.3342 | 0.2797 | 0.2673 |

Headline metric: rg_er (RadGraph partial reward). The baseline run
reproduces the published 0.287, which validates the metric path. Full
write-up in `writeup.md`, including a 16-case manual review and the known
limitations.

## Repository layout

    app.py                  web UI server (FastAPI, port 8000, PORT env)
    index.html              web UI page (no external resources)
    cxr_show.py             terminal tool: browse the 500 results, 16-case
                            review, live predict (.dcm or .png), metrics
    eval_llavarad.py        evaluation script (load, generate, score,
                            metrics.json). the committed results were
                            produced with this exact version
    dcm2png.py              batch DICOM -> 518px PNG (same decode as the
                            training pipeline)
    build_iu_llava.py       data builder: raw IU DICOM -> 518px PNG + report
                            pairs, per-patient train/test split
    run_train_llava.sh      training launch script (auto-resumes)
    llava_iu/train.json     2837 train pairs (image name + report)
    llava_iu/test.json      500 held-out test pairs
    llava_iu/images/        the 500 test images (train images not included,
                            see "What is NOT in this repo")
    baseline_preds.jsonl    saved baseline predictions (500 test ids)
    finetuned_preds.jsonl   saved fine-tuned predictions (500 test ids)
    metrics.json            the numbers in the table above
    review_shortlist.jsonl  the 16 manual-review cases with per-case scores
    writeup.md              full write-up

## Setup (fresh machine, ~16 GB free VRAM)

The project is laid out around three home directories; code paths point at
them, so use this layout:

    ~/iu-xray/                <- clone this repo HERE
    ~/llava-rad-checkpoints/  <- model weights (step 3)
    ~/LLaVA-Rad/              <- official repo, patched (retraining only)

1. clone:

       git clone https://github.com/vijayakumarharsath/llava-rad-iu ~/iu-xray

2. venv (python 3.12):

       python3.12 -m venv ~/llavarad-env
       ~/llavarad-env/bin/pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
       ~/llavarad-env/bin/pip install -r ~/iu-xray/requirements.txt

3. weights, all from Hugging Face:

       ~/llavarad-env/bin/pip install -q "huggingface_hub[cli]"
       huggingface-cli download lmsys/vicuna-7b-v1.5 --local-dir ~/llava-rad-checkpoints/vicuna-7b-v1.5
       huggingface-cli download microsoft/llava-rad --local-dir ~/llava-rad-checkpoints/llavarad
       huggingface-cli download vijayakumarharsath/llava-rad-iu-adapter --local-dir ~/llava-rad-checkpoints/llavarad-iu

   note: the saved config.json in the adapter folder carries a 4-bit
   quantization block on purpose; eval_llavarad.py strips it at load time.

4. only if retraining: clone https://github.com/microsoft/LLaVA-Rad into
   ~/LLaVA-Rad and apply the 3 small compatibility patches described in
   writeup.md (reproducibility section). Not needed to run or evaluate.

## Run it

Web UI (model loads once at startup, ~5 GB VRAM):

    cd ~ && nohup ~/llavarad-env/bin/python ~/iu-xray/app.py > ~/ui.log 2>&1 &

open http://localhost:8000 (or the machine's ip:8000). Browse any of the
500 test cases, click the 16 review cases, or upload your own .dcm/.png for
a live report; DICOM uploads also show the patient/study header data.
Stop: pkill -f "iu-xray/app.py"

Terminal tool:

    ~/llavarad-env/bin/python ~/iu-xray/cxr_show.py

Reproduce the results table (~3 s per image, 500 images):

    CUDA_VISIBLE_DEVICES=1 ~/llavarad-env/bin/python ~/iu-xray/eval_llavarad.py --mode all

## What is NOT in this repo (and why)

- model weights (multi-GB) -> Hugging Face, setup step 3
- the 75 GB raw IU DICOM archive -> it is the public IU collection;
  rebuild with build_iu_llava.py if you need the train images
- the 2837 train images -> rebuildable the same way; only the 500 test
  images are committed, which is all the UI and the eval need
- patient uploads and demo files -> never committed, see .gitignore

## Data notes

- source: open IU chest X-ray collection (NLMCXR). 3955 reports with
  findings; 3337 pair with a usable frontal image (2837 train / 500 test),
  split per patient, no patient appears in both sets
- quirk: ages and numbers are anonymized as the word "ray" in the source
  data, which occasionally leaks into outputs (documented in writeup.md)
