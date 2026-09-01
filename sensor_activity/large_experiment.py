"""Larger repeat of the sensor experiment: 6 channels, 24 time steps."""

import json
import math
import random
from pathlib import Path


CHANNELS, STEPS, INPUTS, LABELS = 6, 24, 144, 4


def softmax(values):
    peak = max(values); values = [math.exp(v - peak) for v in values]; total = sum(values)
    return [v / total for v in values]


def sequence(label, rng):
    phase, scale = rng.uniform(-0.3, 0.3), rng.uniform(0.8, 1.2)
    rows = []
    for channel in range(CHANNELS):
        values = []
        for t in range(STEPS):
            noise = rng.gauss(0, 0.18)
            if label == 0:
                signal = 0.0
            elif label == 1:
                signal = scale * (0.30 + channel * 0.025) * math.sin(t * math.pi / 4 + phase + channel * 0.2)
            elif label == 2:
                signal = scale * (0.45 + channel * 0.035) * math.sin(t * 3 * math.pi / 8 + phase + channel * 0.15)
            else:
                signal = scale * (0.20 + channel * 0.02) * (t / STEPS - 0.5) + 0.25 * math.sin(t * math.pi / 6 + phase + channel)
            values.append(signal + noise)
        rows.append(values)
    return {"inputs": [value for row in rows for value in row], "target": label}


def dataset(path, per_class=100, seed=7, label_noise=0.05):
    rng = random.Random(seed)
    rows = [sequence(label, rng) for label in range(LABELS) for _ in range(per_class)]
    for row in rows:
        if rng.random() < label_noise: row["target"] = rng.randrange(LABELS)
    rng.shuffle(rows); Path(path).write_text(json.dumps(rows) + "\n", encoding="utf-8"); return rows


def new_model(seed=7):
    rng = random.Random(seed)
    matrix = lambda rows, cols: [[rng.uniform(-0.08, 0.08) for _ in range(cols)] for _ in range(rows)]
    return {"architecture": [INPUTS, 16, 12, LABELS], "w1": matrix(16, INPUTS), "b1": [0.0] * 16,
            "w2": matrix(12, 16), "b2": [0.0] * 12, "w3": matrix(LABELS, 12), "b3": [0.0] * LABELS}


def forward(model, inputs):
    h1 = [math.tanh(sum(w * x for w, x in zip(row, inputs)) + b) for row, b in zip(model["w1"], model["b1"])]
    h2 = [math.tanh(sum(w * x for w, x in zip(row, h1)) + b) for row, b in zip(model["w2"], model["b2"])]
    return h1, h2, softmax([sum(w * x for w, x in zip(row, h2)) + b for row, b in zip(model["w3"], model["b3"])])


def accuracy(model, rows):
    return sum(max(range(LABELS), key=lambda i: forward(model, row["inputs"])[2][i]) == row["target"] for row in rows) / len(rows)


def train(rows, epochs=35, rate=0.025, seed=7):
    model, order_rng = new_model(seed), random.Random(seed + 1)
    for _ in range(epochs):
        order = list(range(len(rows))); order_rng.shuffle(order)
        for index in order:
            inputs, target = rows[index]["inputs"], rows[index]["target"]; h1, h2, output = forward(model, inputs)
            e3 = output[:]; e3[target] -= 1; old3 = [r[:] for r in model["w3"]]
            for k in range(LABELS):
                for j in range(12): model["w3"][k][j] -= rate * e3[k] * h2[j]
                model["b3"][k] -= rate * e3[k]
            e2 = [sum(old3[k][j] * e3[k] for k in range(LABELS)) * (1 - h2[j] ** 2) for j in range(12)]; old2 = [r[:] for r in model["w2"]]
            for j in range(12):
                for i in range(16): model["w2"][j][i] -= rate * e2[j] * h1[i]
                model["b2"][j] -= rate * e2[j]
            for i in range(16):
                error = sum(old2[j][i] * e2[j] for j in range(12)) * (1 - h1[i] ** 2)
                for q, value in enumerate(inputs): model["w1"][i][q] -= rate * error * value
                model["b1"][i] -= rate * error
    return model


def features(inputs):
    channels = [inputs[c * STEPS:(c + 1) * STEPS] for c in range(CHANNELS)]
    names, values = ["mean_abs", "rms", "mean_change"], [sum(abs(x) for x in inputs) / INPUTS, math.sqrt(sum(x*x for x in inputs) / INPUTS), 0.0]
    changes = []
    for c, channel in enumerate(channels):
        change = sum(abs(channel[t] - channel[t-1]) for t in range(1, STEPS)) / (STEPS - 1)
        changes.append(change); names += [f"channel_{c}_rms", f"channel_{c}_change"]
        values += [math.sqrt(sum(x*x for x in channel) / STEPS), change]
    values[2] = sum(changes) / CHANNELS
    return names, values


def corr(a, b):
    am, bm = sum(a)/len(a), sum(b)/len(b); numerator = sum((x-am)*(y-bm) for x,y in zip(a,b))
    denominator = math.sqrt(sum((x-am)**2 for x in a) * sum((y-bm)**2 for y in b)); return numerator / denominator if denominator else 0.0


def role_analysis(model, rows):
    names = features(rows[0]["inputs"])[0]; activations = [[forward(model, row["inputs"])[0][i] for row in rows] for i in range(16)]
    candidates = [[features(row["inputs"])[1][j] for row in rows] for j in range(len(names))]
    table = []
    for i, activation in enumerate(activations):
        ranked = sorted(((names[j], corr(activation, candidate)) for j, candidate in enumerate(candidates)), key=lambda x: abs(x[1]), reverse=True)
        table.append({"node": i, "top_roles": [{"name": n, "correlation": v} for n, v in ranked[:4]]})
    scores = {}
    for row in table:
        for candidate in row["top_roles"]: scores[candidate["name"]] = max(scores.get(candidate["name"], 0), abs(candidate["correlation"]))
    selected = [name for name, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]]
    return table, selected


def train_roles(rows, selected, epochs=100, rate=0.08):
    names = features(rows[0]["inputs"])[0]; indices = [names.index(n) for n in selected]
    vectors = [[1.0] + [features(row["inputs"])[1][i] for i in indices] for row in rows]; weights = [[0.0] * len(vectors[0]) for _ in range(LABELS)]
    for _ in range(epochs):
        for row, vector in zip(rows, vectors):
            output = softmax([sum(w*x for w,x in zip(weight, vector)) for weight in weights]); error = output[:]; error[row["target"]] -= 1
            for k in range(LABELS):
                for i, value in enumerate(vector): weights[k][i] -= rate * error[k] * value
    return {"selected_roles": selected, "weights": weights, "parameters": LABELS * len(weights[0])}


def role_accuracy(model, rows):
    names = features(rows[0]["inputs"])[0]; indices = [names.index(n) for n in model["selected_roles"]]
    return sum(max(range(LABELS), key=lambda k: sum(w*x for w,x in zip(model["weights"][k], [1.0] + [features(row["inputs"])[1][i] for i in indices]))) == row["target"] for row in rows) / len(rows)


def run(root):
    train_rows, test_rows, fresh_rows = dataset(root / "large_train.json", seed=7), dataset(root / "large_test.json", seed=19), dataset(root / "large_fresh.json", seed=101)
    baseline = train(train_rows); table, selected = role_analysis(baseline, train_rows); redesigned = train_roles(train_rows, selected)
    result = {"scale": {"channels": CHANNELS, "steps": STEPS, "inputs": INPUTS, "train_examples": len(train_rows)},
              "baseline": {"train": accuracy(baseline, train_rows), "test": accuracy(baseline, test_rows), "fresh": accuracy(baseline, fresh_rows), "parameters": 16*INPUTS+16+12*16+12+LABELS*12+LABELS},
              "node_roles": table, "selected_roles": selected,
              "redesigned": {"train": role_accuracy(redesigned, train_rows), "test": role_accuracy(redesigned, test_rows), "fresh": role_accuracy(redesigned, fresh_rows), "parameters": redesigned["parameters"]}}
    (root / "large_baseline.json").write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8"); (root / "large_redesigned.json").write_text(json.dumps(redesigned, indent=2) + "\n", encoding="utf-8"); (root / "large_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8"); return result


if __name__ == "__main__": print(json.dumps(run(Path(__file__).parent), indent=2))
