"""Multi-architecture raw-input trace with sparse explanations and controls."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiment import CHANNELS, LABELS, VAL_SUBJECTS, accuracy, forward, load_raw, standardize, train
from input_trace_experiment import local_influence, raw_features

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "advanced_trace"
PAIR = (0, 1)


def family(name):
    suffix = name.split(":", 1)[-1]
    if "energy" in suffix:
        return "movement_energy"
    if "autocorrelation" in suffix or "frequency" in suffix:
        return "temporal_periodicity"
    if "correlation" in suffix:
        return "acceleration_rotation_coupling" if "gyro" in name else "cross_channel_correlation"
    if "level" in suffix:
        return "sensor_level"
    if "variation" in suffix or "slope" in suffix:
        return "temporal_change"
    return suffix


def lasso_fit(x, y, alpha, iterations=120):
    mean = x.mean(0)
    scale = x.std(0)
    scale[scale < 1e-10] = 1
    x = (x - mean) / scale
    y_mean = y.mean()
    y = y - y_mean
    beta = np.zeros(x.shape[1])
    norms = np.mean(x * x, 0)
    for _ in range(iterations):
        for j in range(x.shape[1]):
            residual = y - x @ beta + x[:, j] * beta[j]
            rho = np.mean(x[:, j] * residual)
            beta[j] = np.sign(rho) * max(abs(rho) - alpha, 0) / max(norms[j], 1e-12)
    return beta, mean, scale, y_mean


def r2(y, pred):
    return float(1 - np.sum((y - pred) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12))


def lasso_explain(y_fit, y_val, x_fit, x_val, names, iterations=120):
    best = None
    for alpha in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
        beta, mean, scale, y_mean = lasso_fit(x_fit, y_fit, alpha, iterations=iterations)
        pred = ((x_val - mean) / scale) @ beta + y_mean
        score = r2(y_val, pred)
        nonzero = int(np.count_nonzero(beta))
        candidate = (score, -nonzero, alpha, beta, mean, scale)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    score, _, alpha, beta, mean, scale = best
    selected = np.flatnonzero(np.abs(beta) > 1e-8)
    selected = selected[np.argsort(-np.abs(beta[selected]))]
    return {"alpha": alpha, "validation_r2": score, "nonzero_count": int(len(selected)),
            "features": [{"name": names[i], "family": family(names[i]), "coefficient": float(beta[i])} for i in selected]}


def intervene(x, feature_name, seed):
    result = x.copy(); rng = np.random.default_rng(seed)
    channel = CHANNELS.index(feature_name.split(":", 1)[0].split("×")[0])
    kind = family(feature_name)
    if kind == "movement_energy":
        centered = result[:, channel] - result[:, channel].mean(1, keepdims=True)
        result[:, channel] = result[:, channel].mean(1, keepdims=True) + 0.8 * centered
    elif kind == "sensor_level":
        result[:, channel] += 0.2 * result[:, channel].std(1, keepdims=True)
    elif kind == "temporal_periodicity":
        for i in range(len(result)):
            result[i, channel] = result[i, channel, rng.permutation(result.shape[2])]
    else:
        for i in range(len(result)):
            result[i, channel] = np.roll(result[i, channel], int(rng.integers(1, result.shape[2])))
    return result


def margin_shift(model, before_x, after_x):
    before = forward(model, before_x)[2]
    after = forward(model, after_x)[2]
    margin_before = before[:, 0] - before[:, 1]
    margin_after = after[:, 0] - after[:, 1]
    return float(np.mean(np.abs(margin_after - margin_before))), float(np.mean(before.argmax(1) != after.argmax(1)))


def run(seeds=(7, 11, 19, 23, 31), variants=("flat", "wide")):
    train_x, train_y, subjects, test_x, test_y, _ = load_raw()
    train_x, test_x = standardize(train_x, test_x)
    validation = np.isin(subjects, VAL_SUBJECTS); fitting = ~validation
    fit_features, names = raw_features(train_x[fitting])
    val_features, _ = raw_features(train_x[validation])
    OUT.mkdir(parents=True, exist_ok=True)
    runs = []
    family_hits = {}
    for variant in variants:
        for seed in seeds:
            model = train(train_x[fitting], train_y[fitting], seed, variant=variant)
            np.savez(OUT / f"model_{variant}_{seed}.npz", **model)
            h_fit = forward(model, train_x[fitting])[1]; h_val = forward(model, train_x[validation])[1]
            d_out = model["w3"][:, 0] - model["w3"][:, 1]; bias = model["b3"][0] - model["b3"][1]
            score_fit, score_val = h_fit @ d_out + bias, h_val @ d_out + bias
            explanation = lasso_explain(score_fit, score_val, fit_features, val_features, names)
            for item in explanation["features"]:
                family_hits[item["family"]] = family_hits.get(item["family"], 0) + 1
            influence = local_influence(model, test_x).reshape(len(test_x), 9, 128)
            channel_strength = np.abs(influence).mean((0, 2))
            chosen = explanation["features"][0]["name"] if explanation["features"] else "total_acc_x:energy"
            control_candidates = [n for n in names if family(n) == family(chosen) and n.split(":", 1)[0] != chosen.split(":", 1)[0]]
            chosen_channel = CHANNELS.index(chosen.split(":", 1)[0].split("×")[0])
            control = min(control_candidates, key=lambda n: abs(channel_strength[CHANNELS.index(n.split(":", 1)[0].split("×")[0])] - channel_strength[chosen_channel])) if control_candidates else "body_acc_x:energy"
            intervention_change, intervention_flip = margin_shift(model, test_x, intervene(test_x, chosen, seed + 500))
            control_change, control_flip = margin_shift(model, test_x, intervene(test_x, control, seed + 500))
            runs.append({"variant": variant, "seed": seed, "baseline_accuracy": accuracy(model, test_x, test_y),
                         "top_influence_channel": CHANNELS[int(channel_strength.argmax())],
                         "lasso_explanation": explanation, "selected_feature": chosen, "matched_control": control,
                         "intervention_margin_change": intervention_change, "intervention_flip_rate": intervention_flip,
                         "control_margin_change": control_change, "control_flip_rate": control_flip,
                         "mean_abs_influence": float(np.abs(influence).mean()),
                         "class_signed_influence": {LABELS[c]: influence[test_y == c].mean((0, 1)).tolist() for c in PAIR}})
    result = {"protocol": {"pair": "walking vs walking_upstairs", "variants": list(variants), "seeds": list(seeds), "input": "9 x 128 raw values", "trace": "W1 D1 W2 D2 d_out"}, "runs": runs,
              "feature_family_hits": {key: value for key, value in sorted(family_hits.items(), key=lambda item: -item[1])}}
    (OUT / "advanced_trace_results.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    result = run()
    for row in result["runs"]:
        print(row["variant"], row["seed"], row["baseline_accuracy"], row["lasso_explanation"]["nonzero_count"], row["selected_feature"], row["intervention_margin_change"], row["control_margin_change"])
