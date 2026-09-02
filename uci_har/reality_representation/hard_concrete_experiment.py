"""Raw-only bottleneck with a hard-concrete-style L0 gate.

The gate is a deterministic hard-concrete relaxation during optimization:
log-alpha -> stretched sigmoid -> clipped gate. The expected nonzero probability
is penalized, and evaluation reports both expected and hard active coordinates.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiment import VAL_SUBJECTS, load_raw, standardize

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "hard_concrete"
GAMMA, ZETA, BETA = -0.1, 1.1, 2.0


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def relu(x):
    return np.maximum(x, 0.0)


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    p = np.exp(x)
    return p / p.sum(axis=1, keepdims=True)


def gate_values(log_alpha):
    probability = sigmoid(log_alpha - BETA * np.log(-GAMMA / ZETA))
    raw = sigmoid(log_alpha / BETA)
    stretched = raw * (ZETA - GAMMA) + GAMMA
    gate = np.clip(stretched, 0.0, 1.0)
    derivative = np.where((stretched > 0) & (stretched < 1), (ZETA - GAMMA) * raw * (1 - raw) / BETA, 0.0)
    return gate, probability, derivative

def new_model(seed, k, variant):
    rng = np.random.default_rng(seed)
    width = 64 if variant == "flat" else 96
    fan = 9 * 128
    return {"log_alpha": np.full(fan, -2.0),
            "w1": rng.normal(0, np.sqrt(2 / fan), (fan, width)), "b1": np.zeros(width),
            "w2": rng.normal(0, np.sqrt(2 / width), (width, 16)), "b2": np.zeros(16),
            "w3": rng.normal(0, np.sqrt(2 / 16), (16, k)), "b3": np.zeros(k),
            "w4": rng.normal(0, np.sqrt(2 / k), (k, 6)), "b4": np.zeros(6)}


def forward(model, x, cache=False):
    flat = x.reshape(len(x), -1)
    gate, probability, derivative = gate_values(model["log_alpha"])
    gated = flat * gate
    z1 = gated @ model["w1"] + model["b1"]; h1 = relu(z1)
    z2 = h1 @ model["w2"] + model["b2"]; h2 = relu(z2)
    z3 = h2 @ model["w3"] + model["b3"]
    p = softmax(z3 @ model["w4"] + model["b4"])
    if cache:
        return p, (flat, gate, probability, derivative, gated, z1, h1, z2, h2, z3)
    return p


def accuracy(model, x, y):
    return float(np.mean(forward(model, x).argmax(1) == y))


def train_model(x, y, seed, k, variant, l0_penalty, epochs=25):
    model = new_model(seed, k, variant); rng = np.random.default_rng(seed + 1000)
    moments = {name: [np.zeros_like(value), np.zeros_like(value)] for name, value in model.items()}; step = 0
    for _ in range(epochs):
        for idx in np.array_split(rng.permutation(len(x)), max(1, len(x) // 128)):
            p, cache = forward(model, x[idx], cache=True)
            flat, gate, probability, derivative, gated, z1, h1, z2, h2, z3 = cache
            error = p.copy(); error[np.arange(len(idx)), y[idx]] -= 1; error /= len(idx)
            gradients = {"w4": z3.T @ error, "b4": error.sum(0)}
            dz3 = error @ model["w4"].T
            gradients["w3"] = h2.T @ dz3; gradients["b3"] = dz3.sum(0)
            dz2 = (dz3 @ model["w3"].T) * (z2 > 0)
            gradients["w2"] = h1.T @ dz2; gradients["b2"] = dz2.sum(0)
            dz1 = (dz2 @ model["w2"].T) * (z1 > 0)
            gradients["w1"] = gated.T @ dz1; gradients["b1"] = dz1.sum(0)
            dgate = (dz1 @ model["w1"].T * flat).mean(0) * derivative
            dgate += l0_penalty * probability * (1 - probability)
            gradients["log_alpha"] = dgate
            step += 1
            for name, gradient in gradients.items():
                first, second = moments[name]
                first[:] = .9 * first + .1 * gradient; second[:] = .999 * second + .001 * gradient * gradient
                corrected_first = first / (1 - .9 ** step)
                corrected_second = second / (1 - .999 ** step)
                model[name] -= .001 * corrected_first / (np.sqrt(corrected_second) + 1e-8)
    return model


def run(seeds=(7, 11, 19, 23, 31), variants=("flat", "wide"), ks=(4,), penalties=(0.0, 0.03, 0.1, 0.3)):
    train_x, train_y, subjects, test_x, test_y, _ = load_raw(); train_x, test_x = standardize(train_x, test_x)
    fitting = ~np.isin(subjects, VAL_SUBJECTS); OUT.mkdir(parents=True, exist_ok=True); runs = []
    for variant in variants:
        for k in ks:
            for penalty in penalties:
                for seed in seeds:
                    model = train_model(train_x[fitting], train_y[fitting], seed, k, variant, penalty)
                    np.savez(OUT / f"model_{variant}_k{k}_lambda{penalty}_seed{seed}.npz", **model)
                    gate, probability, _ = gate_values(model["log_alpha"])
                    runs.append({"variant": variant, "k": k, "lambda": penalty, "seed": seed, "test_accuracy": accuracy(model, test_x, test_y),
                                 "expected_active_count": float(probability.sum()), "hard_active_count": int(np.sum(gate > 0.5)),
                                 "gate_mean": float(gate.mean()), "gate_p90": float(np.quantile(gate, .9))})
    result = {"input": "raw 9 x 128", "gate": "deterministic hard-concrete relaxation", "runs": runs}
    (OUT / "hard_concrete_results.json").write_text(json.dumps(result, indent=2)); return result


if __name__ == "__main__":
    for row in run()["runs"]:
        print(row["variant"], row["k"], row["lambda"], row["seed"], row["test_accuracy"], row["hard_active_count"])
