import librosa
import numpy as np
from pydub import AudioSegment

SR = 22050          # sample rate
DURATION = 3        # seconds
TARGET_LEN = SR * DURATION

def extract_features(audio_path: str) -> np.ndarray:
    """
    Audio → fixed-length feature vector
    (Member 1 final responsibility)
    """

    # 1️⃣ Load audio (mp3/wav safe)
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_channels(1).set_frame_rate(SR)

    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
   
    # 2️⃣ Normalize (safe)
    samples = samples / (np.max(np.abs(samples)) + 1e-9)
     
    # 3️⃣ Fix length (VERY IMPORTANT) ✅ FIXED
    if len(samples) > TARGET_LEN:
        samples = samples[:TARGET_LEN]
    else:
        samples = np.pad(samples, (0, TARGET_LEN - len(samples)))

    # 4️⃣ Feature extraction
    mfcc = librosa.feature.mfcc(y=samples, sr=SR, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=samples, sr=SR)
    spectral_centroid = librosa.feature.spectral_centroid(y=samples, sr=SR)
    zero_crossing = librosa.feature.zero_crossing_rate(samples)

    # 5️⃣ Aggregate (mean)
    features = np.concatenate([
        np.mean(mfcc, axis=1),              # 13
        np.mean(chroma, axis=1),            # 12
        np.mean(spectral_centroid, axis=1), # 1
        np.mean(zero_crossing, axis=1)      # 1
    ])

    return features


