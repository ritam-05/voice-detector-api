from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import tempfile
import os
import uvicorn
import traceback
import logging
import threading
import gc
import torch

from model_loader import ensure_models

app = FastAPI(title="Voice Detector API")
logger = logging.getLogger("voice_detector")

# --- GLOBALS ---
GPU_LOCK = threading.Lock()

STAGE1 = None
STAGE2_HUMAN = None
STAGE2_AI = None

API_TTA = 2


# =========================
# STARTUP: LOAD MODELS ONCE
# =========================
@app.on_event("startup")
def load_models_on_startup():
    global STAGE1, STAGE2_HUMAN, STAGE2_AI

    try:
        # 1️⃣ Verify model files exist locally
        ensure_models()

        # 2️⃣ Lazy import to avoid Railway double-init
        from final_inference_logic import load_models

        # 3️⃣ Load models into memory ONCE
        STAGE1, STAGE2_HUMAN, STAGE2_AI = load_models()

        try:
            dev = next(STAGE1.parameters()).device
            logger.info("Loaded models on device: %s", str(dev))
        except Exception:
            logger.info("Loaded models on unknown device")

        if STAGE2_AI is None:
            logger.warning("AASIST not loaded. Stage 2 AI verification limited.")

        app.state.models_loaded = True
        app.state.model_load_error = None
        print("Models loaded successfully")

    except Exception as e:
        app.state.models_loaded = False
        app.state.model_load_error = str(e)
        logger.exception("Failed to load models at startup: %s", str(e))


# =========================
# HEALTH CHECK
# =========================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": getattr(app.state, "models_loaded", False)
    }


# =========================
# PREDICTION ENDPOINT
# =========================
@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):

    if not getattr(app.state, "models_loaded", False):
        raise HTTPException(
            status_code=503,
            detail=f"Models not loaded: {app.state.model_load_error or 'unknown error'}"
        )

    suffix = os.path.splitext(file.filename)[1] or ".wav"
    tmp_path = None

    try:
        # Save upload safely
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "wb") as tmp:
            while chunk := await file.read(1024 * 1024):
                tmp.write(chunk)

        with GPU_LOCK:
            with torch.inference_mode():

                # 🔑 Lazy import (critical for Railway stability)
                from final_inference_logic import predict, explain_layman

                label, score, reason, preloaded = predict(
                    tmp_path,
                    stage1=STAGE1,
                    stage2_human=STAGE2_HUMAN,
                    stage2_ai=STAGE2_AI,
                    tta_n=API_TTA
                )

                chunks, full_wav, hc_factors = preloaded

                layman = explain_layman(
                    label,
                    tmp_path,
                    STAGE1,
                    STAGE2_HUMAN,
                    preloaded_chunks=chunks,
                    preloaded_wav=full_wav,
                    preloaded_factors=hc_factors
                )

        expl_obj = layman if isinstance(layman, dict) else {
            "text": layman,
            "confidence": None,
            "factors": []
        }

        return JSONResponse(jsonable_encoder({
            "label": label,
            "score": float(score),
            "reason": reason,
            "explanation": expl_obj
        }))

    except Exception as e:
        logger.exception("Error during prediction: %s", str(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        # Aggressive memory cleanup (4GB GPU safe)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# =========================
# LOCAL RUN (NOT USED BY RAILWAY)
# =========================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
