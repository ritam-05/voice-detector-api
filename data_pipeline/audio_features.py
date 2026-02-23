import librosa
import numpy as np

TARGET_SR = 16000

def normalize_features(f):
    """
    Per-file feature normalization.
    Removes absolute recording scale.
    """
    return (f - np.mean(f)) / (np.std(f) + 1e-9)

def extract_audio_features(path):
    y, sr = librosa.load(path, sr=TARGET_SR, mono=True)

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc = np.mean(mfcc, axis=1)

    # Spectral features
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    features = np.hstack([mfcc, centroid, bandwidth, zcr])

    # CRITICAL STEP
    features = normalize_features(features)

    return features
