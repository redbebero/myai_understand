"""Train classifiers using only selected human-readable feature families."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiment import VAL_SUBJECTS, accuracy, load_raw, standardize
from human_features import extract_human_features

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "advanced_trace" / "family_classifier_results.json"
ORDERS = [
    ["temporal_change"],
    ["temporal_change", "acceleration_rotation_coupling"],
    ["temporal_change", "acceleration_rotation_coupling", "movement_energy"],
    ["temporal_change", "acceleration_rotation_coupling", "movement_energy", "cross_channel_coordination", "temporal_periodicity", "sensor_level"],
]


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def train_linear(x, y, seed, epochs=500, learning_rate=0.08):
    rng = np.random.default_rng(seed)
    classes = int(y.max() + 1)
    weights = rng.normal(0, 0.01, (x.shape[1], classes))
    bias = np.zeros(classes)
    for _ in range(epochs):
        probabilities = softmax(x @ weights + bias)
        probabilities[np.arange(len(y)), y] -= 1
        weights -= learning_rate * (x.T @ probabilities / len(y))
        bias -= learning_rate * probabilities.mean(0)
    return weights, bias


def run(seeds=(7, 11, 19, 23, 31)):
    train_x, train_y, subjects, test_x, test_y, _ = load_raw()
    train_x, test_x = standardize(train_x, test_x)
    validation = np.isin(subjects, VAL_SUBJECTS); fitting = ~validation
    fit_all, _, columns = extract_human_features(train_x[fitting])
    test_all, _, _ = extract_human_features(test_x)
    results = []
    for order in ORDERS:
        cols = np.concatenate([columns[name] for name in order])
        fit_x, test_x_family = fit_all[:, cols], test_all[:, cols]
        mean, scale = fit_x.mean(0), fit_x.std(0)
        scale[scale < 1e-10] = 1
        fit_x = (fit_x - mean) / scale
        test_x_family = (test_x_family - mean) / scale
        accuracies = []
        for seed in seeds:
            weights, bias = train_linear(fit_x, train_y[fitting], seed)
            predictions = np.argmax(test_x_family @ weights + bias, axis=1)
            accuracies.append(float(np.mean(predictions == test_y)))
        results.append({"families": order, "feature_count": int(len(cols)), "test_accuracy_mean": float(np.mean(accuracies)), "test_accuracy_values": accuracies})
    result = {"training_subjects": "official train subjects excluding validation subjects", "evaluation": "held-out UCI-HAR test subjects", "classifier": "linear softmax trained from scratch", "results": results}
    OUT.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    for row in run()["results"]:
        print(len(row["families"]), "+".join(row["families"]), row["test_accuracy_mean"], row["test_accuracy_values"])
