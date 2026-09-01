"""Small, inspectable polynomial replacements for the two-spiral network."""

import json
import math
from pathlib import Path

from train_spiral import evaluate, forward, load_dataset


def powers(degree):
    return [(i, j) for total in range(degree + 1)
            for i in range(total + 1) for j in [total - i]]


def features(inputs, degree):
    x, y = inputs
    return [x ** i * y ** j for i, j in powers(degree)]


def solve_linear(matrix, vector):
    """Gaussian elimination with pivoting; enough for this tiny system."""
    a = [row[:] + [value] for row, value in zip(matrix, vector)]
    for column in range(len(a)):
        pivot = max(range(column, len(a)), key=lambda row: abs(a[row][column]))
        if abs(a[pivot][column]) < 1e-12:
            raise ValueError("singular polynomial system")
        a[column], a[pivot] = a[pivot], a[column]
        scale = a[column][column]
        a[column] = [value / scale for value in a[column]]
        for row in range(len(a)):
            if row == column:
                continue
            scale = a[row][column]
            a[row] = [left - scale * right for left, right in zip(a[row], a[column])]
    return [row[-1] for row in a]


def fit(rows, targets, degree, ridge=1e-6):
    matrix = [features(row["inputs"], degree) for row in rows]
    width = len(matrix[0])
    normal = [[sum(row[i] * row[j] for row in matrix) for j in range(width)] for i in range(width)]
    normal_target = [sum(row[i] * target for row, target in zip(matrix, targets)) for i in range(width)]
    for i in range(width):
        normal[i][i] += ridge
    return {"degree": degree, "powers": powers(degree), "weights": solve_linear(normal, normal_target)}


def score(model, rows):
    predictions = [predict(model, row["inputs"]) for row in rows]
    accuracy = sum((value >= 0.5) == bool(row["target"]) for value, row in zip(predictions, rows)) / len(rows)
    return {"accuracy": accuracy, "predictions": predictions}


def predict(model, inputs):
    value = sum(weight * feature for weight, feature in zip(model["weights"], features(inputs, model["degree"])))
    return 1 / (1 + math.exp(-max(-60.0, min(60.0, value))))


def teacher_targets(model, rows):
    return [logit(forward(model, row["inputs"])[2]) for row in rows]


def logit(probability):
    probability = max(1e-6, min(1 - 1e-6, probability))
    return math.log(probability / (1 - probability))


def fit_all(root):
    train_rows = load_dataset(root / "spiral_train.json")
    test_rows = load_dataset(root / "spiral_test.json")
    neural = json.loads((root / "spiral_model.json").read_text(encoding="utf-8"))
    results = {"neural_baseline": {"train": evaluate(neural, train_rows), "test": evaluate(neural, test_rows)}}
    models = {}
    for target_name, targets in (
        ("direct", [logit(0.05 if row["target"] == 0 else 0.95) for row in train_rows]),
        ("imitate_neural", teacher_targets(neural, train_rows)),
    ):
        models[target_name] = {}
        for degree in range(1, 5):
            model = fit(train_rows, targets, degree)
            models[target_name][str(degree)] = model
            train_score = score(model, train_rows)
            test_score = score(model, test_rows)
            if target_name == "imitate_neural":
                test_score["neural_agreement"] = sum(
                    (value >= 0.5) == (forward(neural, row["inputs"])[2] >= 0.5)
                    for value, row in zip(test_score["predictions"], test_rows)
                ) / len(test_rows)
            results[f"{target_name}_degree_{degree}"] = {
                "parameters": len(model["weights"]),
                "train": {key: value for key, value in train_score.items() if key != "predictions"},
                "test": {key: value for key, value in test_score.items() if key != "predictions"},
            }
    (root / "polynomial_models.json").write_text(json.dumps(models, indent=2) + "\n", encoding="utf-8")
    (root / "polynomial_results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


if __name__ == "__main__":
    root = Path(__file__).parent
    results = fit_all(root)
    for name, result in results.items():
        print(name, result)
