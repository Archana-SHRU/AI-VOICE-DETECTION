from fastapi import FastAPI, Header, HTTPException, Request
import base64
import tempfile
import os
import numpy as np

from app.model import load_model, predict_voice
from app.audio_utils import extract_features
from app.config import API_KEY

app = FastAPI()

model = load_model()


@app.post("/detect")
async def detect_voice(
    request: Request,
    x_api_key: str = Header(...)
):
    # API key check
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    body = await request.json()

    # 🔥 EXACT FIELD TESTER SENDS
    audio_b64 = body.get("audioBase64")

    if not audio_b64:
        raise HTTPException(status_code=422, detail="audioBase64 field missing")

    # Decode base64
    try:
        audio_bytes = base64.b64decode(audio_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 audio")

    # Save temp audio
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_bytes)
            temp_path = f.name
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to save audio")

    # Feature extraction + prediction
    try:
        features = extract_features(temp_path)
        features = np.nan_to_num(features)
        result, confidence = predict_voice(model, features)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass

    return {
        "result": result,
        "confidence": round(float(confidence), 3)
    }
