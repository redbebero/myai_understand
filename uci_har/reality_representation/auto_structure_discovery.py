"""Discover raw channel-time regions and simple expressions without semantic features."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiment import VAL_SUBJECTS, forward, load_raw, standardize, train
from input_trace_experiment import local_influence
from advanced_input_trace import lasso_explain

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "advanced_trace" / "auto_structure_results.json"


def score(model, x):
    logits = forward(model, x)[2]
    return logits[:, 0] - logits[:, 1]


def load_model(path):
    z = np.load(path)
    return {key: z[key] for key in z.files}


def contiguous_regions(strength, quantile=0.85, minimum_length=8):
    threshold = np.quantile(strength, quantile)
    mask = strength >= threshold
    regions = []
    start = None
    for i, active in enumerate(np.r_[mask, False]):
        if active and start is None:
            start = i
        elif not active and start is not None:
            if i - start >= minimum_length:
                regions.append((start, i, float(strength[start:i].mean())))
            start = None
    if not regions:
        width = min(16, len(strength))
        means = np.convolve(strength, np.ones(width) / width, mode="valid")
        start = int(np.argmax(means))
        regions.append((start, start + width, float(means[start])))
    return sorted(regions, key=lambda region: region[2], reverse=True)[:2]


def discover_regions(model, x):
    influence = np.abs(local_influence(model, x).reshape(len(x), 9, 128)).mean(0)
    regions = []
    for channel in range(9):
        for start, end, strength in contiguous_regions(influence[channel]):
            regions.append({"channel": channel, "start": start, "end": end, "strength": strength})
    return sorted(regions, key=lambda region: region["strength"], reverse=True)[:12], influence


def candidate_expressions(x, regions):
    values, names, metadata = [], [], []

    def add(value, name, info):
        values.append(np.asarray(value, dtype=float)); names.append(name); metadata.append(info)

    for region in regions:
        c, a, b = region["channel"], region["start"], region["end"]
        signal = x[:, c, a:b]
        label = f"x[{c},{a}:{b}]"
        add(signal.mean(1), f"mean({label})", {"kind": "unary", "region": region})
        add(np.mean(signal * signal, 1), f"mean(square({label}))", {"kind": "unary", "region": region})
        add(np.mean(np.abs(np.diff(signal, axis=1)), 1), f"mean(abs(diff({label})))", {"kind": "unary", "region": region})
        add(np.ptp(signal, axis=1), f"max({label})-min({label})", {"kind": "unary", "region": region})
        add(np.polyfit(np.arange(b - a), signal.T, 1)[0], f"slope({label})", {"kind": "unary", "region": region})

    for i, left in enumerate(regions):
        for right in regions[i + 1:]:
            a, b = max(left["start"], right["start"]), min(left["end"], right["end"])
            if b - a < 8:
                continue
            x1, x2 = x[:, left["channel"], a:b], x[:, right["channel"], a:b]
            label = f"x[{left['channel']},{a}:{b}]*x[{right['channel']},{a}:{b}]"
            add(np.mean(x1 * x2, 1), f"mean({label})", {"kind": "pair", "left": left, "right": right})
            x1c, x2c = x1 - x1.mean(1, keepdims=True), x2 - x2.mean(1, keepdims=True)
            denom = np.sqrt(np.sum(x1c * x1c, 1) * np.sum(x2c * x2c, 1))
            add(np.divide(np.sum(x1c * x2c, 1), denom, out=np.zeros(len(x)), where=denom > 1e-12), f"corr(x[{left['channel']},{a}:{b}],x[{right['channel']},{a}:{b}])", {"kind": "pair", "left": left, "right": right})
    return np.column_stack(values), tuple(names), metadata


def run(seeds=(7, 11, 19, 23, 31), variants=("flat", "wide")):
    train_x, _, subjects, test_x, _, _ = load_raw()
    train_x, test_x = standardize(train_x, test_x)
    validation = np.isin(subjects, VAL_SUBJECTS); fitting = ~validation
    results = []
    for variant in variants:
        for seed in seeds:
            model = load_model(HERE / "results" / "advanced_trace" / f"model_{variant}_{seed}.npz")
            regions, influence = discover_regions(model, train_x[fitting])
            fit_expr, names, metadata = candidate_expressions(train_x[fitting], regions)
            val_expr, _, _ = candidate_expressions(train_x[validation], regions)
            explanation = lasso_explain(score(model, train_x[fitting]), score(model, train_x[validation]), fit_expr, val_expr, names, iterations=80)
            for feature in explanation["features"]:
                feature["family"] = "auto_discovered_expression"
            results.append({"variant": variant, "seed": seed, "regions": regions, "expressions": explanation, "mean_abs_influence": float(influence.mean())})
    result = {"discovery": "influence-derived regions, then primitive expression search", "no_semantic_feature_families": True, "runs": results}
    OUT.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    for row in run()["runs"]:
        print(row["variant"], row["seed"], row["expressions"]["validation_r2"], row["expressions"]["features"][:3])
