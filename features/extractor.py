from pathlib import Path

import librosa
import numpy as np


def _safe_float(value):
    if value is None:
        return 0.0
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return v if np.isfinite(v) else 0.0


def extract_features(audio_path, sr=22050, n_mfcc=13):
    audio_path = Path(audio_path)
    if not audio_path.exists() or not audio_path.is_file():
        print(f"Skip missing file: {audio_path}")
        return None

    try:
        y, sr = librosa.load(str(audio_path), sr=sr, mono=True)
    except Exception as exc:
        print(f"Skip unreadable file: {audio_path} ({exc.__class__.__name__})")
        return None
    if y.size == 0:
        return None

    duration_sec = float(y.size / sr)

    S = np.abs(librosa.stft(y))
    rms = librosa.feature.rms(S=S)[0]
    mean_energy = _safe_float(np.mean(rms))

    zcr = _safe_float(np.mean(librosa.feature.zero_crossing_rate(y)[0]))

    if rms.size:
        thr = 0.1 * np.max(rms)
        silence_ratio = _safe_float(np.mean(rms < thr)) if thr > 0 else 0.0
        if rms.size > 1:
            times = np.arange(rms.size)
            decay_rate = _safe_float(np.polyfit(times, rms, 1)[0])
        else:
            decay_rate = 0.0
    else:
        silence_ratio = 0.0
        decay_rate = 0.0

    harmonic = librosa.effects.harmonic(y)
    harm_energy = np.sqrt(np.mean(harmonic ** 2)) if harmonic.size else 0.0
    total_energy = np.sqrt(np.mean(y ** 2)) if y.size else 0.0
    harmonicity_ratio = _safe_float(harm_energy / total_energy) if total_energy > 0 else 0.0

    f0 = librosa.yin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
    )
    f0_valid = f0[np.isfinite(f0)]
    f0_mean = _safe_float(np.mean(f0_valid)) if f0_valid.size else 0.0
    f0_median = _safe_float(np.median(f0_valid)) if f0_valid.size else 0.0

    spectral_centroid = _safe_float(np.mean(librosa.feature.spectral_centroid(S=S, sr=sr)[0]))
    bandwidth = _safe_float(np.mean(librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]))

    S_norm = S / (np.sum(S, axis=0, keepdims=True) + 1e-10)
    if S_norm.shape[1] > 1:
        flux = np.sqrt(np.sum(np.diff(S_norm, axis=1) ** 2, axis=0))
        spectral_flux = _safe_float(np.mean(flux))
    else:
        spectral_flux = 0.0

    spectral_roughness = _safe_float(np.mean(librosa.feature.spectral_flatness(S=S)[0]))

    chroma = librosa.feature.chroma_stft(S=S, sr=sr)
    chroma_mean = [float(v) for v in np.mean(chroma, axis=1)]

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = [float(v) for v in np.mean(mfcc, axis=1)]

    feature_vector = [
        mean_energy,
        zcr,
        silence_ratio,
        decay_rate,
        harmonicity_ratio,
        f0_mean,
        f0_median,
        spectral_centroid,
        bandwidth,
        spectral_flux,
        spectral_roughness,
    ] + chroma_mean + mfcc_mean

    return {
        "duration_sec": duration_sec,
        "mean_energy": mean_energy,
        "mfcc": mfcc_mean,
        "zcr": zcr,
        "silence_ratio": silence_ratio,
        "decay_rate": decay_rate,
        "harmonicity_ratio": harmonicity_ratio,
        "f0_mean": f0_mean,
        "f0_median": f0_median,
        "spectral_centroid": spectral_centroid,
        "bandwidth": bandwidth,
        "spectral_flux": spectral_flux,
        "spectral_roughness": spectral_roughness,
        "chroma": chroma_mean,
        "feature_vector": feature_vector,
    }
