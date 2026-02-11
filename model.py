import joblib
import numpy as np

MODEL_PATH = "voice_detector.pkl"


def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")


def predict_voice(model, features: np.ndarray):
    """
    Returns:
        result (str): AI_GENERATED or HUMAN
        confidence (float): 0.0 - 1.0
    """

    # Ensure correct shape for sklearn
    features = features.reshape(1, -1)

    # Predict class
    prediction = model.predict(features)[0]

    # Predict probability (confidence)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[0]
        confidence = max(probs)
    else:
        # Fallback for models without predict_proba
        confidence = 0.5

    result = "AI_GENERATED" if prediction == 1 else "HUMAN"

    return result, confidence
