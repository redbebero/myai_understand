"""Compare human-designed features with AI-discovered raw formulas on unseen data."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from experiment import VAL_SUBJECTS, load_raw, standardize
from family_classifier_evaluation import train_linear

HERE = Path(__file__).resolve().parent
AUTO = HERE / "results" / "bottleneck_gate" / "latent_symbolic_results.json"
HUMAN = HERE / "results" / "advanced_trace" / "family_classifier_results.json"
OUT = HERE / "results" / "advanced_trace" / "representation_comparison.json"
REF = re.compile(r"x\[(\d+),(\d+):(\d+)\]")


def corr(a, b):
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
    den = np.sqrt(np.sum(a*a, 1) * np.sum(b*b, 1))
    return np.divide(np.sum(a*b, 1), den, out=np.zeros(len(a)), where=den > 1e-12)


def evaluate_expression(name, x):
    refs = [(int(c), int(a), int(b)) for c, a, b in REF.findall(name)]
    signals = [x[:, c, a:b] for c, a, b in refs]
    if name.startswith("mean(square("):
        return np.mean(signals[0] ** 2, 1)
    if name.startswith("mean(abs(diff("):
        return np.mean(np.abs(np.diff(signals[0], axis=1)), 1)
    if name.startswith("max("):
        return np.ptp(signals[0], 1)
    if name.startswith("slope("):
        return np.polyfit(np.arange(signals[0].shape[1]), signals[0].T, 1)[0]
    if name.startswith("corr("):
        return corr(signals[0], signals[1])
    if len(signals) == 2:
        width = min(signals[0].shape[1], signals[1].shape[1])
        return np.mean(signals[0][:, :width] * signals[1][:, :width], 1)
    return signals[0].mean(1)


def normalize_fit_test(fit, test):
    mean, scale = fit.mean(0), fit.std(0); scale[scale < 1e-10] = 1
    return (fit - mean) / scale, (test - mean) / scale


def run():
    train_x, train_y, subjects, test_x, test_y, _ = load_raw(); train_x, test_x = standardize(train_x, test_x)
    fitting = ~np.isin(subjects, VAL_SUBJECTS)
    auto = json.loads(AUTO.read_text()); rows = []
    for model in auto["runs"]:
        names = []
        for latent in model["latent_results"]:
            for item in latent["expressions"]["features"][:3]:
                if item["name"] not in names: names.append(item["name"])
        fit = np.column_stack([evaluate_expression(name, train_x[fitting]) for name in names])
        test = np.column_stack([evaluate_expression(name, test_x) for name in names])
        fit, test = normalize_fit_test(fit, test); accuracies = []
        for seed in (7, 11, 19, 23, 31):
            w, b = train_linear(fit, train_y[fitting], seed)
            accuracies.append(float(np.mean(np.argmax(test @ w + b, 1) == test_y)))
        rows.append({"variant": model["variant"], "seed": model["seed"], "expression_count": len(names), "test_accuracy_mean": float(np.mean(accuracies)), "test_accuracy_values": accuracies, "expressions": names})
    human = json.loads(HUMAN.read_text())
    summary = [{"method": "human_top3_families", "test_accuracy": human["results"][2]["test_accuracy_mean"]}, {"method": "human_all6_families", "test_accuracy": human["results"][3]["test_accuracy_mean"]}, {"method": "ai_discovered_raw_expressions", "test_accuracy": float(np.mean([r["test_accuracy_mean"] for r in rows])), "expression_count_mean": float(np.mean([r["expression_count"] for r in rows]))}]
    result = {"training": "fit subjects only", "evaluation": "untouched UCI-HAR test subjects", "summary": summary, "automatic_runs": rows}
    OUT.write_text(json.dumps(result, indent=2)); return result


if __name__ == "__main__":
    for row in run()["summary"]: print(row)
