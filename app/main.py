from fastapi import FastAPI, Header, HTTPException, status
from pydantic import BaseModel
import base64
import tempfile
import os
import numpy as np

from app.model import load_model, predict_voice
from app.audio_utils import extract_features
from app.config import API_KEY

app = FastAPI(
    title="AI Generated Voice Detection API",
    version="1.0"
)

# Load model once at startup
model = load_model()


# ---------- Request & Response Schemas ----------

class AudioRequest(BaseModel):
    audio_base64: str


class DetectionResponse(BaseModel):
    result: str
    confidence: float


# ---------- API Key Validation ----------

def validate_api_key(x_api_key: str):
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )


# ---------- Main Endpoint ----------

@app.post("/detect", response_model=DetectionResponse)
def detect_voice(
    payload: AudioRequest,
    x_api_key: str = Header(...)
):
    validate_api_key(x_api_key)

    # Decode Base64 audio
    try:
        audio_bytes = base64.b64decode(payload.audio_base64)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid base64 audio data"
        )

    # Save to temp file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(audio_bytes)
            temp_path = tmp.name
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Failed to create temporary audio file"
        )

    try:
        # Feature extraction
        features = extract_features(temp_path)
        features = np.nan_to_num(features)

        # Prediction
        result, confidence = predict_voice(model, features)

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported or corrupted audio: {str(e)}"
        )

    finally:
        # Safe cleanup (Windows-friendly)
        try:
            os.remove(temp_path)
        except PermissionError:
            pass

    return {
        "result": result,
        "confidence": round(float(confidence), 3)
    }
