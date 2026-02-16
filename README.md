# 🎙 AI Generated Voice Detection API

This project detects whether a given voice sample is **AI-generated** or **Human** using audio feature extraction and a machine learning classifier.

---

## 🚀 Overview

The API accepts a **Base64-encoded MP3 audio file** and returns:

- Classification result (AI / Human)
- Confidence score (0.0 – 1.0)

Supported languages:
- Tamil
- English
- Hindi
- Malayalam
- Telugu

---

## 🧠 How It Works

1. User sends Base64 audio to the API.
2. Audio is decoded and converted to mono (fixed sample rate).
3. Audio features (MFCC, Chroma, Spectral Centroid, Zero Crossing Rate) are extracted.
4. A trained ML model classifies the voice.
5. JSON response is returned.

---

## 🛠 Tech Stack

- Python
- FastAPI
- Librosa
- Pydub
- Scikit-learn
- Joblib
- Uvicorn

---

## 📡 API Endpoint

### POST `/detect`

### Input (JSON)

```json
{
  "audio_base64": "BASE64_ENCODED_AUDIO"
}
