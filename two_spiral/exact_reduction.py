"""Search for exact algebraic redundancy in the trained two-spiral network."""

import copy
import json
import math
from pathlib import Path

from train_spiral import evaluate, forward, load_dataset, nonzero_weights, prune_connection, prune_neuron


WEIGHT_KEYS = ("weights_input_hidden1", "weights_hidden1_hidden2", "weights_hidden2_output")


def rank(matrix, tolerance=1e-10):
    a = [row[:] for row in matrix]
    rows, columns, result = len(a), len(a[0]), 0
    for column in range(columns):
        pivot = max(range(result, rows), key=lambda row: abs(a[row][column]), default=result)
        if pivot >= rows or abs(a[pivot][column]) <= tolerance:
            continue
        a[result], a[pivot] = a[pivot], a[result]
        scale = a[result][column]
        a[result] = [value / scale for value in a[result]]
        for row in range(rows):
            if row != result:
                scale = a[row][column]
                a[row] = [left - scale * right for left, right in zip(a[row], a[result])]
        result += 1
    return result


def logit(probability):
    probability = max(1e-12, min(1 - 1e-12, probability))
    return math.log(probability / (1 - probability))


def probe_points(rows):
    points = [row["inputs"] for row in rows]
    for index in range(41):
        value = -1 + 2 * index / 40
        for other in range(41):
            points.append([value, -1 + 2 * other / 40])
    return points


def output(model, point):
    return logit(forward(model, point)[2])


def compare(original, candidate, points):
    differences = [abs(output(original, point) - output(candidate, point)) for point in points]
    agreements = sum((output(original, point) >= 0) == (output(candidate, point) >= 0) for point in points)
    return {"max_logit_difference": max(differences), "mean_logit_difference": sum(differences) / len(differences), "decision_agreement": agreements / len(points)}


def duplicate_pairs(model, layer, tolerance=1e-12):
    if layer == 1:
        rows, biases = model["weights_input_hidden1"], model["bias_hidden1"]
    else:
        rows, biases = model["weights_hidden1_hidden2"], model["bias_hidden2"]
    return [[i, j] for i in range(len(rows)) for j in range(i + 1, len(rows))
            if max(abs(a - b) for a, b in zip(rows[i], rows[j])) <= tolerance
            and abs(biases[i] - biases[j]) <= tolerance]


def merge_duplicate(model, layer, first, second):
    candidate = copy.deepcopy(model)
    if layer == 1:
        for row in candidate["weights_hidden1_hidden2"]:
            row[first] += row[second]
        candidate["weights_input_hidden1"][second] = [0.0, 0.0]
        candidate["bias_hidden1"][second] = 0.0
        for row in candidate["weights_hidden1_hidden2"]:
            row[second] = 0.0
    else:
        candidate["weights_hidden2_output"][first] += candidate["weights_hidden2_output"][second]
        candidate["weights_hidden2_output"][second] = 0.0
        candidate["weights_hidden1_hidden2"][second] = [0.0] * len(candidate["weights_hidden1_hidden2"][second])
        candidate["bias_hidden2"][second] = 0.0
    return candidate


def run(root):
    model = json.loads((root / "spiral_model.json").read_text(encoding="utf-8"))
    train_rows = load_dataset(root / "spiral_train.json")
    test_rows = load_dataset(root / "spiral_test.json")
    points = probe_points(test_rows)
    result = {
        "architecture": model["architecture"],
        "parameter_count": 205,
        "nonzero_weights": nonzero_weights(model),
        "zero_weight_counts": {key: sum(value == 0.0 for value in (item for row in model[key] for item in (row if isinstance(row, list) else [row]))) for key in WEIGHT_KEYS},
        "matrix_ranks": {
            "input_hidden1": rank(model["weights_input_hidden1"]),
            "hidden1_hidden2": rank(model["weights_hidden1_hidden2"]),
            "hidden2_output": rank([model["weights_hidden2_output"]]),
        },
        "duplicate_neurons": {"hidden1": duplicate_pairs(model, 1), "hidden2": duplicate_pairs(model, 2)},
        "single_connection_removals_preserving_grid": [],
        "single_neuron_removals": [],
        "exact_duplicate_merges": [],
    }
    for layer, rows, columns in ((0, 12, 2), (1, 12, 12), (2, 1, 12)):
        for row in range(rows):
            for column in range(columns):
                candidate = copy.deepcopy(model)
                prune_connection(candidate, layer, row, column)
                comparison = compare(model, candidate, points)
                if comparison["max_logit_difference"] <= 1e-12:
                    result["single_connection_removals_preserving_grid"].append([layer, row, column])
    for layer in (1, 2):
        for index in range(12):
            candidate = copy.deepcopy(model)
            prune_neuron(candidate, layer, index)
            comparison = compare(model, candidate, points)
            result["single_neuron_removals"].append({"layer": layer, "index": index, **comparison, "test_accuracy": evaluate(candidate, test_rows)})
    for layer in (1, 2):
        for first, second in duplicate_pairs(model, layer):
            candidate = merge_duplicate(model, layer, first, second)
            result["exact_duplicate_merges"].append({"layer": layer, "neurons": [first, second], **compare(model, candidate, points)})
    (root / "exact_reduction_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(Path(__file__).parent), indent=2))
