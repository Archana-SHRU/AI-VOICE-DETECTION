import librosa
import numpy as np
from pydub import AudioSegment


def extract_features(audio_path: str) -> np.ndarray:
    """
    Converts audio to mono WAV, extracts features,
    and returns a fixed-length feature vector.
    """

    # Load audio using pydub (handles mp3 safely)
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_channels(1).set_frame_rate(22050)

    # Convert to raw samples
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)

    # Normalize
    samples = samples / np.max(np.abs(samples))

    # Extract features using librosa
    mfcc = librosa.feature.mfcc(y=samples, sr=22050, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=samples, sr=22050)
    spectral_centroid = librosa.feature.spectral_centroid(y=samples, sr=22050)
    zero_crossing = librosa.feature.zero_crossing_rate(samples)

    # Aggregate (mean)
    features = np.concatenate([
        np.mean(mfcc, axis=1),
        np.mean(chroma, axis=1),
        np.mean(spectral_centroid, axis=1),
        np.mean(zero_crossing, axis=1)
    ])

    return features
