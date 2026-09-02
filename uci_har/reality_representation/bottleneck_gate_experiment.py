"""Raw-only learnable gate plus bottleneck experiment.

No human-designed sensor features are used. The model learns both which raw
coordinates to retain and how many latent coordinates are needed for activity
classification.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiment import VAL_SUBJECTS, accuracy, load_raw, standardize

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "bottleneck_gate"


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def relu(x):
    return np.maximum(x, 0.0)
def model_accuracy(model, x, y):
    return float(np.mean(forward(model, x).argmax(1) == y))



def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    p = np.exp(x)
    return p / p.sum(axis=1, keepdims=True)


def new_model(seed, k, variant):
    rng = np.random.default_rng(seed)
    width = 64 if variant == "flat" else 96
    fan = 9 * 128
    return {
        "gate_logits": np.full(fan, 2.0),
        "w1": rng.normal(0, np.sqrt(2 / fan), (fan, width)), "b1": np.zeros(width),
        "w2": rng.normal(0, np.sqrt(2 / width), (width, 16)), "b2": np.zeros(16),
        "w3": rng.normal(0, np.sqrt(2 / 16), (16, k)), "b3": np.zeros(k),
        "w4": rng.normal(0, np.sqrt(2 / k), (k, 6)), "b4": np.zeros(6),
    }


def forward(model, x, training=False):
    flat = x.reshape(len(x), -1)
    gate = sigmoid(model["gate_logits"])
    gated = flat * gate
    z1 = gated @ model["w1"] + model["b1"]
    h1 = relu(z1)
    z2 = h1 @ model["w2"] + model["b2"]
    h2 = relu(z2)
    z3 = h2 @ model["w3"] + model["b3"]
    probabilities = softmax(z3 @ model["w4"] + model["b4"])
    if training:
        return probabilities, (flat, gate, gated, z1, h1, z2, h2, z3)
    return probabilities


def train_model(x, y, seed, k, variant, gate_penalty=0.003, epochs=25, batch_size=128):
    model = new_model(seed, k, variant)
    rng = np.random.default_rng(seed + 1000)
    moments = {name: [np.zeros_like(value), np.zeros_like(value)] for name, value in model.items()}
    step = 0
    for _ in range(epochs):
        for idx in np.array_split(rng.permutation(len(x)), max(1, len(x) // batch_size)):
            probabilities, cache = forward(model, x[idx], training=True)
            flat, gate, gated, z1, h1, z2, h2, z3 = cache
            error = probabilities.copy()
            error[np.arange(len(idx)), y[idx]] -= 1
            error /= len(idx)
            gradients = {}
            gradients["w4"] = z3.T @ error; gradients["b4"] = error.sum(0)
            dz3 = error @ model["w4"].T
            gradients["w3"] = h2.T @ dz3; gradients["b3"] = dz3.sum(0)
            dh2 = dz3 @ model["w3"].T
            dz2 = dh2 * (z2 > 0)
            gradients["w2"] = h1.T @ dz2; gradients["b2"] = dz2.sum(0)
            dh1 = dz2 @ model["w2"].T
            dz1 = dh1 * (z1 > 0)
            gradients["w1"] = gated.T @ dz1; gradients["b1"] = dz1.sum(0)
            d_gated = dz1 @ model["w1"].T
            d_gate = (d_gated * flat).mean(0) + gate_penalty / len(gate) * gate * (1 - gate)
            gradients["gate_logits"] = d_gate
            step += 1
            for name, gradient in gradients.items():
                first, second = moments[name]
                first[:] = 0.9 * first + 0.1 * gradient
                second[:] = 0.999 * second + 0.001 * gradient * gradient
                corrected_first = first / (1 - 0.9 ** step)
                corrected_second = second / (1 - 0.999 ** step)
                model[name] -= 0.001 * corrected_first / (np.sqrt(corrected_second) + 1e-8)
    return model


def run(seeds=(7, 11, 19, 23, 31), variants=("flat", "wide"), ks=(16, 8, 4, 2, 1)):
    train_x, train_y, subjects, test_x, test_y, _ = load_raw()
    train_x, test_x = standardize(train_x, test_x)
    validation = np.isin(subjects, VAL_SUBJECTS); fitting = ~validation
    OUT.mkdir(parents=True, exist_ok=True)
    runs = []
    for variant in variants:
        for k in ks:
            for seed in seeds:
                model = train_model(train_x[fitting], train_y[fitting], seed, k, variant, gate_penalty=0.3)
                np.savez(OUT / f"model_{variant}_k{k}_seed{seed}.npz", **model)
                gate = sigmoid(model["gate_logits"])
                runs.append({"variant": variant, "k": k, "seed": seed,
                             "test_accuracy": model_accuracy(model, test_x, test_y),
                             "validation_accuracy": float(np.mean(forward(model, train_x[validation]).argmax(1) == train_y[validation])),
                             "active_gate_fraction_05": float(np.mean(gate > 0.5)),
                             "active_gate_fraction_08": float(np.mean(gate > 0.8)),
                             "mean_gate": float(gate.mean())})
    result = {"input": "raw 9 x 128 only", "gate_penalty": 0.3, "runs": runs}
    (OUT / "bottleneck_gate_results.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    result = run()
    for r in result["runs"]:
        print(r["variant"], r["k"], r["seed"], r["test_accuracy"], r["active_gate_fraction_05"])
