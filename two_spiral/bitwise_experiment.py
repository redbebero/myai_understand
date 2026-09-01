"""Measure how much of the two-spiral model survives low-bit arithmetic."""

import json
import math
from pathlib import Path

from train_spiral import evaluate, forward, load_dataset


WEIGHT_KEYS = ("weights_input_hidden1", "weights_hidden1_hidden2", "weights_hidden2_output")
BIAS_KEYS = ("bias_hidden1", "bias_hidden2", "bias_output")


def flatten(values):
    if not isinstance(values, list):
        return [values]
    return [item for value in values for item in flatten(value)]


def quantize(values, bits):
    limit = 2 ** (bits - 1) - 1
    scale = max(abs(value) for value in flatten(values)) / limit or 1.0

    def convert(value):
        return max(-limit, min(limit, round(value / scale))) * scale

    if isinstance(values, list):
        return [quantize(value, bits) if isinstance(value, list) else convert(value) for value in values]
    return convert(values)


def quantized_model(model, bits):
    result = dict(model)
    for key in WEIGHT_KEYS + BIAS_KEYS:
        result[key] = quantize(model[key], bits)
    return result


def signed_model(model):
    result = dict(model)
    for key in WEIGHT_KEYS:
        scale = mean_abs(model[key])
        result[key] = [[sign(value) * scale for value in row] for row in model[key]] if isinstance(model[key][0], list) else [sign(value) * scale for value in model[key]]
    return result


def sign(value):
    return 1 if value >= 0 else -1


def binary_dot(left, right):
    """XNOR + popcount equivalent: equal signs add, unequal signs subtract."""
    matches = sum(sign(a) == sign(b) for a, b in zip(left, right))
    return 2 * matches - len(left)


def mean_abs(values):
    return sum(abs(value) for value in flatten(values)) / len(flatten(values)) or 1.0


def binary_forward(model, inputs):
    weight1 = model["weights_input_hidden1"]
    weight2 = model["weights_hidden1_hidden2"]
    weight3 = model["weights_hidden2_output"]
    scale1, scale2, scale3 = map(mean_abs, (weight1, weight2, weight3))
    input_bits = [sign(value) for value in inputs]
    hidden1 = [sign(scale1 * binary_dot(row, input_bits) + bias) for row, bias in zip(weight1, model["bias_hidden1"])]
    hidden2 = [sign(scale2 * binary_dot(row, hidden1) + bias) for row, bias in zip(weight2, model["bias_hidden2"])]
    output = scale3 * binary_dot(weight3, hidden2) + model["bias_output"]
    return 1 / (1 + math.exp(-max(-60.0, min(60.0, output))))


def evaluate_binary(model, rows):
    return sum((binary_forward(model, row["inputs"]) >= 0.5) == bool(row["target"]) for row in rows) / len(rows)


def evaluate_variants(root):
    rows_train = load_dataset(root / "spiral_train.json")
    rows_test = load_dataset(root / "spiral_test.json")
    model = json.loads((root / "spiral_model.json").read_text(encoding="utf-8"))
    results = {
        "original": {"train": evaluate(model, rows_train), "test": evaluate(model, rows_test)},
        "binary_weights": {"train": evaluate(signed_model(model), rows_train), "test": evaluate(signed_model(model), rows_test)},
        "binary_network": {"train": evaluate_binary(model, rows_train), "test": evaluate_binary(model, rows_test)},
    }
    for bits in (8, 4, 2):
        candidate = quantized_model(model, bits)
        results[f"{bits}_bit"] = {"train": evaluate(candidate, rows_train), "test": evaluate(candidate, rows_test)}
    (root / "bitwise_results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


if __name__ == "__main__":
    for name, result in evaluate_variants(Path(__file__).parent).items():
        print(name, result)
