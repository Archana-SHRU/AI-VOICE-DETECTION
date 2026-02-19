# 🎙 AI Generated Voice Detection API

## 🧩 Problem Statement

With the rapid advancement of AI voice synthesis technologies, it has become increasingly difficult to distinguish between **human voices** and **AI-generated voices**.

This creates serious concerns in:

- Deepfake voice scams
- Fake political speeches
- Identity fraud
- Misinformation spread
- Voice-based authentication systems

There is a need for a reliable system that can automatically detect whether a voice sample is real or AI-generated.

---

## 💡 Proposed Solution

We built a REST API that analyzes an audio sample and determines whether the voice is:

- 🎤 Human
- 🤖 AI-Generated

The system works by:

1. Accepting a Base64-encoded MP3 audio file.
2. Converting audio to a standardized format (mono, fixed duration).
3. Extracting key audio features (MFCC, Chroma, Spectral Centroid, Zero Crossing Rate).
4. Using a trained Machine Learning classifier to predict authenticity.
5. Returning a JSON response with:
   - Result (AI / Human)
   - Confidence Score (0.0 – 1.0)

---

## 🌍 Supported Languages

The model is designed to work across multiple Indian languages:

- Tamil
- English
- Hindi
- Malayalam
- Telugu

---

## 🚀 Who Can Use This?

This system can be useful for:

- 🏦 Banks & Financial Institutions (prevent voice fraud)
- 📰 Media Platforms (detect fake audio clips)
- 🏛 Government Agencies (deepfake detection)
- 📞 Call Centers (voice authentication validation)
- 🔐 Cybersecurity Teams (audio threat analysis)
- 🎓 Research & AI Ethics Studies

---

## 🛠 Tech Stack

- Python
- FastAPI
- Librosa (Audio Processing)
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

```json
{
  "audio_base64": "BASE64_ENCODED_AUDIO"
}
output:

{
  "result": "AI",
  "confidence": 0.87
}

**PROJECT STRUTURE**
app/
 ├── __init__.py
 ├── main.py
 ├── model.py
 ├── audio_utils.py

requirements.txt
voice_detector.pkl
README.md

**

**pip install -r requirements.txt
uvicorn app.main:app --reload
**


📌 Conclusion**
This project demonstrates a practical AI-driven approach to tackling the growing threat of AI-generated voice deepfakes across multiple languages.

It combines audio signal processing and machine learning to provide a scalable and deployable detection system.
