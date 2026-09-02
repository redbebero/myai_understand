"""Greedy human-readable family compression of the walking/upstairs score."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from experiment import VAL_SUBJECTS, forward, load_raw, standardize
from input_trace_experiment import raw_features
from advanced_input_trace import lasso_explain

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "advanced_trace" / "family_compression_results.json"
ORDER_FAMILIES = ("movement_energy", "temporal_periodicity", "temporal_change", "acceleration_rotation_coupling", "sensor_level", "cross_channel_coordination")


def family(name):
    suffix = name.split(":", 1)[-1]
    if "energy" in suffix:
        return "movement_energy"
    if "autocorrelation" in suffix or "frequency" in suffix:
        return "temporal_periodicity"
    if any(token in suffix for token in ("variation", "slope", "peak")):
        return "temporal_change"
    if "correlation" in suffix:
        return "acceleration_rotation_coupling" if "gyro" in name else "cross_channel_coordination"
    if "level" in suffix:
        return "sensor_level"
    return "other"


def score(model, x):
    out = forward(model, x)[2]
    return out[:, 0] - out[:, 1]


def load_model(path):
    z = np.load(path)
    return {key: z[key] for key in z.files}


def run(seeds=(7, 11, 19, 23, 31), variants=("flat", "wide")):
    train_x, _, subjects, test_x, _, _ = load_raw()
    train_x, _ = standardize(train_x, test_x)
    validation = np.isin(subjects, VAL_SUBJECTS); fitting = ~validation
    x_fit, names = raw_features(train_x[fitting])
    x_val, _ = raw_features(train_x[validation])
    indices = {name: np.array([i for i, item in enumerate(names) if family(item) == name]) for name in ORDER_FAMILIES}
    all_results = []
    for variant in variants:
        for seed in seeds:
            model = load_model(HERE / "results" / "advanced_trace" / f"model_{variant}_{seed}.npz")
            y_fit = score(model, train_x[fitting]); y_val = score(model, train_x[validation])
            selected, remaining, curve = [], list(ORDER_FAMILIES), []
            for step in range(len(ORDER_FAMILIES)):
                candidates = []
                for candidate in remaining:
                    trial = selected + [candidate]
                    cols = np.concatenate([indices[item] for item in trial])
                    explanation = lasso_explain(y_fit, y_val, x_fit[:, cols], x_val[:, cols], tuple(names[i] for i in cols), iterations=20)
                    candidates.append((explanation["validation_r2"], candidate, explanation))
                best_r2, best_family, best_explanation = max(candidates, key=lambda row: row[0])
                selected.append(best_family); remaining.remove(best_family)
                curve.append({"step": step + 1, "selected_families": list(selected), "validation_r2": best_r2,
                              "nonzero_count": best_explanation["nonzero_count"], "selected_features": best_explanation["features"]})
            all_results.append({"variant": variant, "seed": seed, "curve": curve})
    result = {"families": list(ORDER_FAMILIES), "selection": "greedy by held-out validation R2", "runs": all_results}
    OUT.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    result = run()
    for row in result["runs"]:
        print(row["variant"], row["seed"], " -> ".join(row["curve"][-1]["selected_families"]))
        print("  ", [(step["step"], round(step["validation_r2"], 3), step["nonzero_count"]) for step in row["curve"]])
