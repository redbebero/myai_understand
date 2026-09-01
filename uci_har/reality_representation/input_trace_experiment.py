"""Trace walking-vs-upstairs decisions from raw input to human features."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiment import CHANNELS, LABELS, ROOT, VAL_SUBJECTS, accuracy, forward, load_raw, new_model, standardize, train

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "input_trace"
PAIR = (0, 1)


def save_model(model, path):
    np.savez(path, **model)


def local_influence(model, x):
    h1, h2, _ = forward(model, x)
    d1, d2 = h1 > 0, h2 > 0
    d_out = model["w3"][:, PAIR[0]] - model["w3"][:, PAIR[1]]
    values = []
    for row_d1, row_d2 in zip(d1, d2):
        values.append((model["w1"] * row_d1[None, :]) @ model["w2"] @ (row_d2 * d_out))
    return np.asarray(values)


def raw_features(x):
    n, channels, timesteps = x.shape
    features, names = [], []
    freqs = np.fft.rfftfreq(timesteps, d=1 / 50)
    for c, channel in enumerate(CHANNELS):
        s = x[:, c]
        centered = s - s.mean(1, keepdims=True)
        spectrum = np.abs(np.fft.rfft(centered, axis=1))
        nonzero = spectrum[:, 1:]
        dom = freqs[1:][np.argmax(nonzero, axis=1)]
        ac = np.array([np.corrcoef(row[:-1], row[1:])[0, 1] for row in s])
        for name, value in (
            ("level", s.mean(1)), ("variation", np.mean(np.abs(np.diff(s, axis=1)), 1)),
            ("energy", np.mean(s * s, 1)), ("slope", np.polyfit(np.arange(timesteps), s.T, 1)[0]),
            ("peak", s.max(1)), ("peak_to_peak", np.ptp(s, axis=1)), ("dominant_frequency", dom),
            ("autocorrelation_lag1", ac),
        ):
            features.append(value); names.append(f"{channel}:{name}")
    for a in range(channels):
        for b in range(a + 1, channels):
            xa, xb = x[:, a], x[:, b]
            corr = np.array([np.corrcoef(u, v)[0, 1] for u, v in zip(xa, xb)])
            lag_values, lag_names = [], []
            for lag in range(-8, 9):
                if lag < 0:
                    lag_values.append(np.array([np.corrcoef(u[:lag], v[-lag:])[0, 1] for u, v in zip(xa, xb)]))
                elif lag > 0:
                    lag_values.append(np.array([np.corrcoef(u[lag:], v[:-lag])[0, 1] for u, v in zip(xa, xb)]))
                else:
                    lag_values.append(corr)
                lag_names.append(lag)
            lag_values = np.asarray(lag_values).T
            features += [corr, np.max(np.abs(lag_values), 1)]
            names += [f"{CHANNELS[a]}×{CHANNELS[b]}:correlation", f"{CHANNELS[a]}×{CHANNELS[b]}:lagged_correlation"]
    return np.asarray(features).T, tuple(names)


def standard_feature_matrix(values):
    mean, scale = values.mean(0), values.std(0)
    scale[scale < 1e-10] = 1
    return (values - mean) / scale


def fit_explanation(score_fit, score_val, features_fit, features_val, names):
    fit_mean, fit_scale = features_fit.mean(0), features_fit.std(0)
    fit_scale[fit_scale < 1e-10] = 1
    fit_z = (features_fit - fit_mean) / fit_scale
    val_z = (features_val - fit_mean) / fit_scale
    center = score_fit.mean()
    centered = score_fit - center
    corr = np.array([np.corrcoef(centered, fit_z[:, i])[0, 1] for i in range(fit_z.shape[1])])
    order = np.argsort(np.abs(corr))[::-1]
    selected = order[:12]
    coef, *_ = np.linalg.lstsq(fit_z[:, selected], centered, rcond=None)
    pred = val_z[:, selected] @ coef + center
    r2 = 1 - np.sum((pred - score_val) ** 2) / max(np.sum((score_val - score_val.mean()) ** 2), 1e-12)
    return {"top_features": [{"name": names[i], "fit_correlation": float(corr[i])} for i in selected],
            "sparse_like_coefficients": [{"name": names[i], "coefficient": float(c)} for i, c in zip(selected, coef)],
            "validation_r2": float(r2)}


def circular_shift(x, channel, seed):
    result = x.copy(); rng = np.random.default_rng(seed)
    for i in range(len(result)):
        result[i, channel] = np.roll(result[i, channel], int(rng.integers(1, x.shape[2])))
    return result


def run(seeds=(7, 11, 19, 23, 31)):
    train_x, train_y, subjects, test_x, test_y, test_subjects = load_raw()
    train_x, test_x = standardize(train_x, test_x)
    validation = np.isin(subjects, VAL_SUBJECTS); fitting = ~validation
    fit_features, feature_names = raw_features(train_x[fitting])
    val_features, _ = raw_features(train_x[validation])
    OUT.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in seeds:
        model = train(train_x[fitting], train_y[fitting], seed, variant="flat")
        save_model(model, OUT / f"model_seed_{seed}.npz")
        h2_fit = forward(model, train_x[fitting])[1]
        h2_val = forward(model, train_x[validation])[1]
        h2_test = forward(model, test_x)[1]
        d_out = model["w3"][:, PAIR[0]] - model["w3"][:, PAIR[1]]
        bias_diff = model["b3"][PAIR[0]] - model["b3"][PAIR[1]]
        score_fit = h2_fit @ d_out + bias_diff
        score_val = h2_val @ d_out + bias_diff
        score_test = h2_test @ d_out + bias_diff
        influence = local_influence(model, test_x).reshape(len(test_x), 9, 128)
        class_maps = {LABELS[c]: {"mean_signed": influence[test_y == c].mean(0).tolist(), "mean_absolute": np.abs(influence[test_y == c]).mean(0).tolist()} for c in PAIR}
        top_channel = np.abs(influence).mean(0).sum(1).argmax()
        explanation = fit_explanation(score_fit, score_val, fit_features, val_features, feature_names)
        best = explanation["top_features"][0]["name"]
        channel = CHANNELS.index(best.split(":")[0].split("×")[0])
        control_channel = (channel + 1) % len(CHANNELS)
        baseline = accuracy(model, test_x, test_y)
        intervention = accuracy(model, circular_shift(test_x, channel, seed + 100), test_y)
        control = accuracy(model, circular_shift(test_x, control_channel, seed + 100), test_y)
        runs.append({"seed": seed, "baseline_accuracy": baseline,
                     "score_fit_mean": float(score_fit.mean()), "score_val_mean": float(score_val.mean()),
                     "score_test_mean": float(score_test.mean()), "top_influence_channel": CHANNELS[top_channel],
                     "class_influence_maps": class_maps, "direct_score_explanation": explanation,
                     "selected_intervention_feature": best, "intervention_channel": CHANNELS[channel],
                     "intervention_accuracy": intervention, "intervention_drop": baseline - intervention,
                     "matched_control_channel": CHANNELS[control_channel],
                     "matched_control_accuracy": control, "matched_control_drop": baseline - control})
    result = {"protocol": {"pair": "walking vs walking_upstairs", "seeds": list(seeds), "input": "9 x 128 raw sensor values", "influence": "samplewise ReLU-gated local Jacobian"}, "runs": runs}
    (OUT / "input_trace_results.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    result = run()
    for row in result["runs"]:
        print(row["seed"], row["baseline_accuracy"], row["top_influence_channel"], row["selected_intervention_feature"], row["intervention_drop"])
