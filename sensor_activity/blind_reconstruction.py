"""Reconstruct roles using only sensor files and a trained model.

This file deliberately does not import the dataset generator or use its labels to
choose formulas. It proposes generic time-series statistics, compares them with
hidden activations, and learns only a final classifier over selected statistics.
"""

import json
import math
from pathlib import Path


CHANNELS, STEPS, LABELS = 3, 12, 4


def softmax(values):
    peak = max(values)
    exp_values = [math.exp(value - peak) for value in values]
    total = sum(exp_values)
    return [value / total for value in exp_values]


def forward(model, inputs):
    h1 = [math.tanh(sum(w * x for w, x in zip(row, inputs)) + bias) for row, bias in zip(model["w1"], model["b1"])]
    h2 = [math.tanh(sum(w * x for w, x in zip(row, h1)) + bias) for row, bias in zip(model["w2"], model["b2"])]
    logits = [sum(w * x for w, x in zip(row, h2)) + bias for row, bias in zip(model["w3"], model["b3"])]
    return h1, h2, softmax(logits)


def predict(model, inputs):
    return max(range(LABELS), key=lambda label: forward(model, inputs)[2][label])


def accuracy(model, rows):
    return sum(predict(model, row["inputs"]) == row["target"] for row in rows) / len(rows)


def candidate_features(inputs):
    channels = [inputs[c * STEPS:(c + 1) * STEPS] for c in range(CHANNELS)]
    changes = [channel[t] - channel[t - 1] for channel in channels for t in range(1, STEPS)]
    names = ["mean_abs", "rms", "mean_abs_change", "trend", "channel_0_mean", "channel_1_mean", "channel_2_mean",
             "channel_0_rms", "channel_1_rms", "channel_2_rms", "channel_0_change", "channel_1_change", "channel_2_change"]
    values = [
        sum(abs(value) for value in inputs) / len(inputs),
        math.sqrt(sum(value * value for value in inputs) / len(inputs)),
        sum(abs(value) for value in changes) / len(changes),
        sum(channels[0][-1] - channels[0][0] for _ in [0]) / STEPS,
        *[sum(channel) / STEPS for channel in channels],
        *[math.sqrt(sum(value * value for value in channel) / STEPS) for channel in channels],
        *[sum(abs(channel[t] - channel[t - 1]) for t in range(1, STEPS)) / (STEPS - 1) for channel in channels],
    ]
    return names, values


def correlations(model, rows):
    names, _ = candidate_features(rows[0]["inputs"])
    activations = [[forward(model, row["inputs"])[0][node] for row in rows] for node in range(len(model["w1"]))]
    candidates = [[candidate_features(row["inputs"])[1][feature] for row in rows] for feature in range(len(names))]

    def corr(left, right):
        left_mean, right_mean = sum(left) / len(left), sum(right) / len(right)
        numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
        denominator = math.sqrt(sum((a - left_mean) ** 2 for a in left) * sum((b - right_mean) ** 2 for b in right))
        return numerator / denominator if denominator else 0.0

    table = []
    for node, values in enumerate(activations):
        ranked = sorted(((names[index], corr(values, candidate)) for index, candidate in enumerate(candidates)), key=lambda item: abs(item[1]), reverse=True)
        table.append({"node": node, "top_candidate_roles": [{"name": name, "correlation": value} for name, value in ranked[:4]]})
    return names, table


def select_roles(table, count=7):
    scores = {}
    for row in table:
        for candidate in row["top_candidate_roles"]:
            scores[candidate["name"]] = max(scores.get(candidate["name"], 0.0), abs(candidate["correlation"]))
    return [name for name, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:count]]


def train_role_model(rows, selected, epochs=300, learning_rate=0.08):
    names = candidate_features(rows[0]["inputs"])[0]
    indices = [names.index(name) for name in selected]
    features = [[1.0] + [candidate_features(row["inputs"])[1][index] for index in indices] for row in rows]
    weights = [[0.0] * len(features[0]) for _ in range(LABELS)]
    for _ in range(epochs):
        for row, values in zip(rows, features):
            probabilities = softmax([sum(w * x for w, x in zip(weight, values)) for weight in weights])
            error = probabilities[:]
            error[row["target"]] -= 1.0
            for label in range(LABELS):
                for index, value in enumerate(values):
                    weights[label][index] -= learning_rate * error[label] * value
    return {"selected_roles": selected, "weights": weights, "parameter_count": LABELS * len(weights[0])}


def role_predict(model, inputs):
    names, values = candidate_features(inputs)
    selected = model["selected_roles"]
    vector = [1.0] + [values[names.index(name)] for name in selected]
    return max(range(LABELS), key=lambda label: sum(w * x for w, x in zip(model["weights"][label], vector)))


def role_accuracy(model, rows):
    return sum(role_predict(model, row["inputs"]) == row["target"] for row in rows) / len(rows)


def run(root):
    model = json.loads((root / "sensor_baseline.json").read_text(encoding="utf-8"))
    train_rows = json.loads((root / "sensor_train.json").read_text(encoding="utf-8"))
    test_rows = json.loads((root / "sensor_test.json").read_text(encoding="utf-8"))
    fresh_rows = json.loads((root / "sensor_fresh.json").read_text(encoding="utf-8"))
    _, table = correlations(model, train_rows)
    selected = select_roles(table)
    redesigned = train_role_model(train_rows, selected)
    results = {
        "input_sources": ["sensor_baseline.json", "sensor_train.json", "sensor_test.json", "sensor_fresh.json"],
        "generator_used": False,
        "node_role_correlations": table,
        "selected_roles": selected,
        "baseline": {"train": accuracy(model, train_rows), "test": accuracy(model, test_rows), "fresh": accuracy(model, fresh_rows)},
        "blind_redesign": {"train": role_accuracy(redesigned, train_rows), "test": role_accuracy(redesigned, test_rows), "fresh": role_accuracy(redesigned, fresh_rows), "parameters": redesigned["parameter_count"]},
    }
    (root / "blind_reconstruction_model.json").write_text(json.dumps(redesigned, indent=2) + "\n", encoding="utf-8")
    (root / "blind_reconstruction_results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


if __name__ == "__main__":
    print(json.dumps(run(Path(__file__).parent), indent=2))
