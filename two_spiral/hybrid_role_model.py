"""Human-designed nodes, with only their final combination weights learned."""

import json
import math
from pathlib import Path

from human_role_model import K, evaluate as fixed_evaluate, nodes


def role_features(inputs):
    groups = nodes(inputs)
    return [
        1.0,
        groups["distance"]["center_confidence"],
        groups["distance"]["outer_region"],
        groups["direction"]["x_positive"],
        groups["direction"]["y_positive"],
        groups["spiral"]["arm_0"],
        groups["spiral"]["arm_1"],
    ]


def sigmoid(value):
    return 1 / (1 + math.exp(-max(-60.0, min(60.0, value))))


def predict_probability(model, inputs):
    return sigmoid(sum(weight * value for weight, value in zip(model["combination_weights"], role_features(inputs))))


def predict(model, inputs):
    return int(predict_probability(model, inputs) >= 0.5)


def evaluate(model, rows):
    return sum(predict(model, row["inputs"]) == row["target"] for row in rows) / len(rows)


def train(rows, epochs=3000, learning_rate=0.2):
    # Only these seven output weights change. K and every role-node equation stay fixed.
    weights = [0.0] * len(role_features(rows[0]["inputs"]))
    for _ in range(epochs):
        for row in rows:
            features = role_features(row["inputs"])
            error = sigmoid(sum(w * x for w, x in zip(weights, features))) - row["target"]
            for index, feature in enumerate(features):
                weights[index] -= learning_rate * error * feature
    return {
        "architecture": "fixed distance/direction/spiral nodes -> learned logistic vote",
        "fixed_spiral_constant": K,
        "combination_weights": weights,
        "trainable_parameter_count": len(weights),
    }


def run(root):
    train_rows = json.loads((root / "spiral_train.json").read_text(encoding="utf-8"))
    test_rows = json.loads((root / "spiral_test.json").read_text(encoding="utf-8"))
    model = train(train_rows)
    results = {
        "fixed_nodes": {
            "role_equations_learned": False,
            "combination_weights_learned": False,
            "train_accuracy": fixed_evaluate(train_rows),
            "test_accuracy": fixed_evaluate(test_rows),
        },
        "hybrid": {
            "role_equations_learned": False,
            "combination_weights_learned": True,
            "trainable_parameter_count": model["trainable_parameter_count"],
            "train_accuracy": evaluate(model, train_rows),
            "test_accuracy": evaluate(model, test_rows),
            "learned_combination_weights": model["combination_weights"],
        },
    }
    (root / "hybrid_role_model.json").write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    (root / "hybrid_role_results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


if __name__ == "__main__":
    for name, result in run(Path(__file__).parent).items():
        print(name, result)
