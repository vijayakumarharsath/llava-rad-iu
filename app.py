#!/usr/bin/env python3
# app.py -- browser UI for the LLaVA-Rad (IU chest X-ray) project
#
# Run:
#   ~/llavarad-env/bin/python ~/iu-xray/app.py
# Then open in a browser (from another Tailscale machine):
#   http://100.96.182.65:8000
#
# The fine-tuned model is loaded ONCE at startup (~5 GB VRAM) using the
# SAME verified load path as the eval (eval_llavarad.load_model). DICOM
# decoding and generation reuse cxr_show (which itself is a verbatim copy
# of the official eval mechanics). Stop with: pkill -f "iu-xray/app.py"
#
# DRAFT REPORTS FOR HUMAN REVIEW -- NEVER A DIAGNOSIS.

import base64
import io
import json
import os
import sys
import tempfile
import threading
import time
from datetime import date

HOME = os.path.expanduser("~")
IU = os.path.join(HOME, "iu-xray")
PORT = int(os.environ.get("PORT", "8000"))

sys.path.insert(0, IU)
sys.path.insert(0, os.path.join(HOME, "LLaVA-Rad"))

import eval_llavarad as ev        # verified load path + data files
import cxr_show                   # _open_image (dcm/png) + _predict + data

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
import uvicorn

app = FastAPI()
data = cxr_show.load_all()
MODEL = None
MODEL_LOCK = threading.Lock()  # guards the one-time load
GEN_LOCK = threading.Lock()    # serializes model.generate calls

TRAIN_LOG = os.path.join(HOME, "train_resume3.log")
TRAIN_ROWS = [
    ("base model", "LLaVA-Rad: Vicuna-7B LLM + BiomedCLIP-CXR 518 encoder"),
    ("method", "QLoRA: 4-bit NF4 double-quant + LoRA r=64 / alpha=128 / dropout 0.05"),
    ("trained", "LoRA adapters (q,k,v,o,gate,up,down) + image->text projector"),
    ("frozen", "vision encoder + base LLM"),
    ("data", "2837 IU X-ray image-report pairs (train split, per patient)"),
    ("epochs", "3"),
    ("iterations", "531 total = 177 per epoch (2837 / effective batch 16)"),
    ("batch", "effective 16 (1 image x 16 gradient accumulation, 1 GPU)"),
    ("lr", "1e-4, cosine schedule, 3% warmup, weight decay 0"),
    ("precision", "bf16 + TF32, gradient checkpointing, max seq len 2048"),
    ("speed", "~51 s per iteration"),
    ("final run", "231 iterations (step 300 -> 531) in 11800 s = ~3 h 17 m"),
    ("final loss", "0.212 at epoch 3.0"),
]


def get_model():
    """Load once, on first call (or at startup via the preloader)."""
    global MODEL
    with MODEL_LOCK:
        if MODEL is None:
            t0 = time.time()
            model, tokenizer, iproc = ev.load_model(ev.ADAPTER_DIR, "cuda")
            print("app: model ready in %.0fs" % (time.time() - t0), flush=True)
            MODEL = {"model": model, "tokenizer": tokenizer, "iproc": iproc}
    return MODEL


def dcm_meta(path):
    """Read the DICOM header (patient/study data) from an uploaded file.
    CPU-only, never touches the pixels (stop_before_pixels), never raises:
    returns a list of [label, value] rows (non-empty fields only) or None
    for non-DICOM files / unreadable headers. Rows only go to the user's
    own browser over Tailscale; nothing is logged or saved."""
    if path.lower().endswith((".png", ".jpg", ".jpeg")):
        return None
    try:
        import pydicom
        ds = pydicom.dcmread(path, stop_before_pixels=True)
    except Exception:
        return None
    rows = []
    def add(label, value):
        value = str(value).strip() if value is not None else ""
        if value:
            rows.append([label, value])
    add("patient", ds.get("PatientName"))
    add("patient id", ds.get("PatientID"))
    sex = str(ds.get("PatientSex") or "").strip()
    if sex:
        rows.append(["sex", {"M": "male", "F": "female", "O": "other"}.get(sex, sex)])
    birth = str(ds.get("PatientBirthDate") or "").strip()
    if birth:
        rows.append(["birth date", birth])
    study = str(ds.get("StudyDate") or "").strip()
    if study:
        rows.append(["study date", study])
    age = ""
    if len(birth) == 8 and len(study) == 8:
        try:
            b = date(int(birth[:4]), int(birth[4:6]), int(birth[6:]))
            s = date(int(study[:4]), int(study[4:6]), int(study[6:]))
            days = (s - b).days
            if days >= 0:
                age = "%d years" % int(days // 365.25)
        except Exception:
            age = ""
    if not age:
        raw = str(ds.get("PatientAge") or "").strip()  # DICOM form, e.g. 045Y
        if raw and raw[0].isdigit():
            unit = {"Y": "years", "M": "months", "W": "weeks",
                    "D": "days"}.get(raw[-1])
            if unit:
                age = "%d %s" % (int(raw[:-1]), unit)
    if age:
        rows.append(["age at scan", age])
    add("study", ds.get("StudyDescription"))
    add("series", ds.get("SeriesDescription"))
    add("modality", ds.get("Modality"))
    add("body part", ds.get("BodyPartExamined"))
    add("institution", ds.get("InstitutionName"))
    add("referring physician", ds.get("ReferringPhysicianName"))
    add("accession", ds.get("AccessionNumber"))
    return rows or None


# ------------------------------ pages ------------------------------------

@app.get("/")
def index():
    return FileResponse(os.path.join(IU, "index.html"), media_type="text/html")


@app.get("/img/{fname}")
def img(fname):
    fname = os.path.basename(fname)  # no path traversal
    p = os.path.join(ev.IMAGE_DIR, fname)
    if not os.path.exists(p):
        return JSONResponse({"error": "image not found"}, 404)
    return FileResponse(p, media_type="image/png")


# ------------------------------ data api ---------------------------------

@app.get("/api/status")
def status():
    return {"ready": MODEL is not None}


@app.get("/api/bootstrap")
def bootstrap():
    metrics = {}
    if os.path.exists(ev.METRICS_JSON):
        with open(ev.METRICS_JSON) as f:
            metrics = json.load(f)
    train_line = ""
    if os.path.exists(TRAIN_LOG):
        with open(TRAIN_LOG, errors="replace") as f:
            for ln in f:
                if "train_runtime" in ln:
                    train_line = ln.strip()
    return {
        "ids": data["test_ids"],
        "review": data["short"],
        "metrics": metrics,
        "train": {"rows": [list(r) for r in TRAIN_ROWS], "log_line": train_line},
        "counts": {"base": len(data["base"]), "ft": len(data["ft"])},
    }


@app.get("/api/case/{rid}")
def case(rid):
    resolved, cands = cxr_show.resolve_id(rid, data)
    if resolved is None:
        return JSONResponse(
            {"error": "unknown id", "candidates": (cands or [])[:10]}, 404)
    b = data["base"].get(resolved)
    if b is None:
        return JSONResponse({"error": "unknown id: %s" % rid}, 404)
    t = data["test_map"].get(resolved, {})
    f = data["ft"].get(resolved)
    out = {
        "id": resolved,
        "image": t.get("image", ""),
        "query": b["query"].replace("<image>", "").strip(),
        "reference": b["reference"],
        "baseline": b["prediction"],
        "finetuned": f["prediction"] if f else None,
    }
    s = next((x for x in data["short"] if x["id"] == resolved), None)
    if s:
        out["review"] = {k: s[k] for k in s
                         if k in ("reason",) or
                         (k.startswith(("baseline_", "finetuned_", "delta_"))
                          and isinstance(s[k], (int, float)))}
    return out


# ------------------------------ predict ----------------------------------

@app.post("/api/predict")
async def predict(rid: str = Form(""), file: UploadFile = File(None)):
    tmp_path = None
    try:
        image_b64 = None
        image_out = None
        result_id = None
        reference = None
        dmeta = None
        if rid and rid.strip():
            s = rid.strip()
            resolved, cands = cxr_show.resolve_id(s, data)
            if resolved is None:
                return JSONResponse(
                    {"error": "unknown id", "candidates": (cands or [])[:10]}, 400)
            result_id = resolved
            image_path = os.path.join(
                ev.IMAGE_DIR, data["test_map"][resolved]["image"])
            query = (data["base"][resolved]["query"]
                     if resolved in data["base"] else cxr_show.DEFAULT_QUERY)
            image_out = "/img/" + data["test_map"][resolved]["image"]
            if resolved in data["base"]:
                reference = data["base"][resolved]["reference"]
        elif file is not None:
            os.makedirs(os.path.join(IU, "uploads"), exist_ok=True)
            suffix = os.path.splitext(file.filename or "upload.png")[1].lower()
            tmp = tempfile.NamedTemporaryFile(
                suffix=suffix, delete=False, dir=os.path.join(IU, "uploads"))
            tmp.write(await file.read())
            tmp.close()
            tmp_path = tmp.name
            image_path = tmp_path
            dmeta = dcm_meta(tmp_path)  # patient/study data from the header
            query = cxr_show.DEFAULT_QUERY
        else:
            return JSONResponse({"error": "provide an id or a file"}, 400)

        image = cxr_show._open_image(image_path)  # CPU-only decode
        try:
            m = get_model()
        except Exception as e:
            return JSONResponse({"error": "model not ready: %s" % e}, 503)
        with GEN_LOCK:  # one generate at a time
            pred, dt = cxr_show._predict(m["model"], m["tokenizer"],
                                         m["iproc"], image, query)
        if file is not None:
            buf = io.BytesIO()
            image.convert("RGB").save(buf, "PNG")
            image_b64 = base64.b64encode(buf.getvalue()).decode()
        return {
            "prediction": pred,
            "seconds": round(dt, 2),
            "id": result_id,
            "image": image_out,
            "image_b64": image_b64,
            "dcm_meta": dmeta,
            "reference": reference,
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == "__main__":
    # Preload the model in the background so the page can open immediately.
    threading.Thread(target=get_model, daemon=True).start()
    print("app: serving on 0.0.0.0:%d (open http://<ip>:%d)" % (PORT, PORT),
          flush=True)
    uvicorn.run(app, host="0.0.0.0", port=PORT)
