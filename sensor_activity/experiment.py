"""Sensor activity experiment: train, inspect, ablate, and redesign without learning roles."""

import json
import math
import random
from pathlib import Path


LABELS = ("idle", "walk", "run", "turn")
CHANNELS = 3
STEPS = 12
INPUTS = CHANNELS * STEPS


def softmax(values):
    peak = max(values)
    exps = [math.exp(value - peak) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def make_sequence(label, rng):
    values = [[0.0] * STEPS for _ in range(CHANNELS)]
    for t in range(STEPS):
        noise = [rng.gauss(0, 0.16) for _ in range(CHANNELS)]
        phase = rng.uniform(-0.25, 0.25)
        scale = rng.uniform(0.85, 1.15)
        if label == 0:  # quiet sensor
            signal = [0.0, 0.0, 0.0]
        elif label == 1:  # steady periodic movement
            signal = [scale * 0.42 * math.sin(t * math.pi / 2 + phase), scale * 0.30 * math.cos(t * math.pi / 2 + phase), 0.08]
        elif label == 2:  # faster, larger periodic movement
            signal = [scale * 0.58 * math.sin(t * 3 * math.pi / 4 + phase), scale * 0.40 * math.cos(t * 3 * math.pi / 4 + phase), 0.15]
        else:  # turn: rotation changes while acceleration stays moderate
            signal = [0.12 * t / STEPS, scale * 0.46 * (t / STEPS - 0.5), scale * 0.42 * math.sin(t * math.pi / 3 + phase)]
        for channel in range(CHANNELS):
            values[channel][t] = signal[channel] + noise[channel]
    return {"inputs": [values[c][t] for c in range(CHANNELS) for t in range(STEPS)], "target": label}


def make_dataset(path, per_class=100, seed=7, label_noise=0.05):
    rng = random.Random(seed)
    rows = [make_sequence(label, rng) for label in range(len(LABELS)) for _ in range(per_class)]
    for row in rows:
        if rng.random() < label_noise:
            row["target"] = rng.randrange(len(LABELS))
    rng.shuffle(rows)
    Path(path).write_text(json.dumps(rows) + "\n", encoding="utf-8")
    return rows


def new_model(seed=7, hidden1=4, hidden2=4):
    rng = random.Random(seed)
    def matrix(rows, columns):
        return [[rng.uniform(-0.15, 0.15) for _ in range(columns)] for _ in range(rows)]
    return {"architecture": [INPUTS, hidden1, hidden2, len(LABELS)],
            "w1": matrix(hidden1, INPUTS), "b1": [0.0] * hidden1,
            "w2": matrix(hidden2, hidden1), "b2": [0.0] * hidden2,
            "w3": matrix(len(LABELS), hidden2), "b3": [0.0] * len(LABELS)}


def forward(model, inputs):
    h1 = [math.tanh(sum(w * x for w, x in zip(row, inputs)) + bias) for row, bias in zip(model["w1"], model["b1"])]
    h2 = [math.tanh(sum(w * x for w, x in zip(row, h1)) + bias) for row, bias in zip(model["w2"], model["b2"])]
    output = softmax([sum(w * x for w, x in zip(row, h2)) + bias for row, bias in zip(model["w3"], model["b3"])])
    return h1, h2, output


def predict(model, inputs):
    return max(range(len(LABELS)), key=lambda i: forward(model, inputs)[2][i])


def evaluate(model, rows):
    return sum(predict(model, row["inputs"]) == row["target"] for row in rows) / len(rows)


def train(rows, epochs=180, learning_rate=0.035, seed=7):
    model = new_model(seed)
    rng = random.Random(seed + 1)
    for _ in range(epochs):
        order = list(range(len(rows)))
        rng.shuffle(order)
        for index in order:
            inputs, target = rows[index]["inputs"], rows[index]["target"]
            h1, h2, output = forward(model, inputs)
            e3 = output[:]
            e3[target] -= 1.0
            old_w3 = [row[:] for row in model["w3"]]
            for k in range(len(LABELS)):
                for j in range(len(h2)):
                    model["w3"][k][j] -= learning_rate * e3[k] * h2[j]
                model["b3"][k] -= learning_rate * e3[k]
            e2 = [sum(old_w3[k][j] * e3[k] for k in range(len(LABELS))) * (1 - h2[j] ** 2) for j in range(len(h2))]
            old_w2 = [row[:] for row in model["w2"]]
            for j, error in enumerate(e2):
                for i in range(len(h1)):
                    model["w2"][j][i] -= learning_rate * error * h1[i]
                model["b2"][j] -= learning_rate * error
            for i in range(len(h1)):
                error = sum(old_w2[j][i] * e2[j] for j in range(len(h2))) * (1 - h1[i] ** 2)
                for q, value in enumerate(inputs):
                    model["w1"][i][q] -= learning_rate * error * value
                model["b1"][i] -= learning_rate * error
    return model


def feature_names():
    return ["mean_abs", "rms", "mean_step_change", "alternating_change", "channel_0_mean", "channel_1_mean", "channel_2_mean"]


def role_features(inputs):
    channels = [inputs[c * STEPS:(c + 1) * STEPS] for c in range(CHANNELS)]
    flat_abs = [abs(value) for value in inputs]
    changes = [channels[c][t] - channels[c][t - 1] for c in range(CHANNELS) for t in range(1, STEPS)]
    alternating = sum(abs(changes[i] - changes[i - 1]) for i in range(1, len(changes))) / max(1, len(changes) - 1)
    return [sum(flat_abs) / len(flat_abs), math.sqrt(sum(value * value for value in inputs) / len(inputs)),
            sum(abs(value) for value in changes) / len(changes), alternating,
            *[sum(channel) / STEPS for channel in channels]]


def redesign_features(rows):
    return [[1.0] + role_features(row["inputs"]) for row in rows]


def train_redesigned(rows, epochs=300, learning_rate=0.08):
    features = redesign_features(rows)
    weights = [[0.0] * len(features[0]) for _ in LABELS]
    for _ in range(epochs):
        for row, values in zip(rows, features):
            probabilities = softmax([sum(w * x for w, x in zip(weight, values)) for weight in weights])
            error = probabilities[:]
            error[row["target"]] -= 1.0
            for label in range(len(LABELS)):
                for i, value in enumerate(values):
                    weights[label][i] -= learning_rate * error[label] * value
    return {"architecture": "fixed sensor-role features -> learned softmax vote", "feature_names": ["bias"] + feature_names(), "weights": weights}


def redesigned_predict(model, inputs):
    values = [1.0] + role_features(inputs)
    return max(range(len(LABELS)), key=lambda label: sum(w * x for w, x in zip(model["weights"][label], values)))


def redesigned_evaluate(model, rows):
    return sum(redesigned_predict(model, row["inputs"]) == row["target"] for row in rows) / len(rows)


def analyze_nodes(model, rows):
    activations = [[forward(model, row["inputs"])[0][i] for row in rows] for i in range(len(model["w1"]))]
    means = [sum(values) / len(values) for values in activations]
    importance = []
    baseline = evaluate(model, rows)
    for index in range(len(model["w1"])):
        candidate = json.loads(json.dumps(model))
        candidate["w1"][index] = [0.0] * INPUTS
        candidate["b1"][index] = 0.0
        candidate["w2"] = [[0.0 if i == index else value for i, value in enumerate(row)] for row in candidate["w2"]]
        importance.append({"node": index, "mean_activation": means[index], "accuracy_after_removal": evaluate(candidate, rows), "accuracy_drop": baseline - evaluate(candidate, rows)})
    return sorted(importance, key=lambda item: item["accuracy_drop"], reverse=True)


def run(root):
    train_rows = make_dataset(root / "sensor_train.json", seed=7)
    test_rows = make_dataset(root / "sensor_test.json", seed=19)
    fresh_rows = make_dataset(root / "sensor_fresh.json", seed=101)
    baseline = train(train_rows)
    redesigned = train_redesigned(train_rows)
    results = {
        "labels": LABELS,
        "baseline": {"train": evaluate(baseline, train_rows), "test": evaluate(baseline, test_rows), "fresh": evaluate(baseline, fresh_rows), "parameters": 4 * INPUTS + 4 + 4 * 4 + 4 + 4 * 4 + 4},
        "node_analysis": analyze_nodes(baseline, train_rows),
        "redesigned": {"train": redesigned_evaluate(redesigned, train_rows), "test": redesigned_evaluate(redesigned, test_rows), "fresh": redesigned_evaluate(redesigned, fresh_rows), "parameters": 4 * 8},
        "redesign_features": feature_names(),
    }
    (root / "sensor_baseline.json").write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
    (root / "sensor_redesigned.json").write_text(json.dumps(redesigned, indent=2) + "\n", encoding="utf-8")
    (root / "sensor_results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


if __name__ == "__main__":
    print(json.dumps(run(Path(__file__).parent), indent=2))
