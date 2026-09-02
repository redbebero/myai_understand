"""Explicit raw-window features used for human-readable score compression.

Input shape: (samples, 9, 128), already standardized exactly as the MLP input.
Every returned array has shape (samples, n_features). No hidden representation is used.
"""
from __future__ import annotations

import numpy as np

from experiment import CHANNELS


FAMILY_NAMES = (
    "movement_energy",
    "sensor_level",
    "temporal_change",
    "temporal_periodicity",
    "cross_channel_coordination",
    "acceleration_rotation_coupling",
)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson correlation per sample; constant windows become zero."""
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.sum(a * a, axis=1) * np.sum(b * b, axis=1))
    numerator = np.sum(a * b, axis=1)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12)


def _lagged_corr(a: np.ndarray, b: np.ndarray, max_lag: int = 8) -> np.ndarray:
    """Maximum absolute correlation over integer lags in [-max_lag, max_lag]."""
    correlations = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            correlations.append(_safe_corr(a[:, :lag], b[:, -lag:]))
        elif lag > 0:
            correlations.append(_safe_corr(a[:, lag:], b[:, :-lag]))
        else:
            correlations.append(_safe_corr(a, b))
    return np.max(np.abs(np.stack(correlations, axis=1)), axis=1)


def extract_human_features(x: np.ndarray, sample_rate: float = 50.0):
    """Return explicit family arrays, feature names, and family column indices.

    Parameters
    ----------
    x:
        Standardized UCI-HAR windows, shape ``(n_samples, 9, 128)``.
    sample_rate:
        UCI-HAR sampling rate in Hz.
    """
    if x.ndim != 3 or x.shape[1] != len(CHANNELS) or x.shape[2] != 128:
        raise ValueError(f"expected (samples, {len(CHANNELS)}, 128), got {x.shape}")

    n_samples, _, n_timesteps = x.shape
    t = np.arange(n_timesteps, dtype=float)
    frequency = np.fft.rfftfreq(n_timesteps, d=1.0 / sample_rate)
    values = []
    names = []
    families = []

    def add(family: str, name: str, value: np.ndarray):
        values.append(np.asarray(value, dtype=float))
        names.append(name)
        families.append(family)

    # 1. Movement energy: mean squared signal in each channel.
    for c, channel in enumerate(CHANNELS):
        signal = x[:, c]
        add("movement_energy", f"{channel}:energy", np.mean(signal ** 2, axis=1))

    # 2. Sensor level: mean signal value in each channel.
    for c, channel in enumerate(CHANNELS):
        add("sensor_level", f"{channel}:level", x[:, c].mean(axis=1))

    # 3. Temporal change: local variation, linear slope, extrema range.
    for c, channel in enumerate(CHANNELS):
        signal = x[:, c]
        slope = np.polyfit(t, signal.T, 1)[0]
        add("temporal_change", f"{channel}:variation", np.mean(np.abs(np.diff(signal, axis=1)), axis=1))
        add("temporal_change", f"{channel}:slope", slope)
        add("temporal_change", f"{channel}:peak", np.max(signal, axis=1))
        add("temporal_change", f"{channel}:peak_to_peak", np.ptp(signal, axis=1))

    # 4. Temporal periodicity: dominant non-DC frequency and lag-1 autocorrelation.
    for c, channel in enumerate(CHANNELS):
        signal = x[:, c]
        centered = signal - signal.mean(axis=1, keepdims=True)
        spectrum = np.abs(np.fft.rfft(centered, axis=1))
        dominant = frequency[1:][np.argmax(spectrum[:, 1:], axis=1)]
        lag1 = _safe_corr(signal[:, :-1], signal[:, 1:])
        add("temporal_periodicity", f"{channel}:dominant_frequency", dominant)
        add("temporal_periodicity", f"{channel}:autocorrelation_lag1", lag1)

    # 5–6. Cross-channel coordination and acceleration–rotation coupling.
    for a in range(len(CHANNELS)):
        for b in range(a + 1, len(CHANNELS)):
            left, right = CHANNELS[a], CHANNELS[b]
            pair = f"{left}×{right}"
            pair_family = (
                "acceleration_rotation_coupling"
                if (("gyro" in left and "acc" in right) or ("gyro" in right and "acc" in left))
                else "cross_channel_coordination"
            )
            corr = _safe_corr(x[:, a], x[:, b])
            lagged = _lagged_corr(x[:, a], x[:, b])
            add(pair_family, f"{pair}:correlation", corr)
            add(pair_family, f"{pair}:lagged_correlation", lagged)

    matrix = np.column_stack(values)
    family_columns = {
        family: np.array([i for i, value in enumerate(families) if value == family], dtype=int)
        for family in FAMILY_NAMES
    }
    return matrix, tuple(names), family_columns


def extract_family_values(x: np.ndarray, sample_rate: float = 50.0):
    """Return ``{family: matrix}`` for direct inspection or downstream modeling."""
    matrix, names, columns = extract_human_features(x, sample_rate)
    return {family: matrix[:, indices] for family, indices in columns.items()}, names
