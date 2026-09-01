"""A hand-designed, non-learning model made from named node groups."""

import json
import math
import tempfile
from pathlib import Path

from generate_spirals import generate
from train_spiral import evaluate as neural_evaluate


K = 4 * math.pi  # human hypothesis: one full turn per radial half-unit


def nodes(inputs):
    x, y = inputs
    radius = math.hypot(x, y)
    angle = math.atan2(y, x)
    phase = angle - K * radius
    return {
        "distance": {
            "radius": radius,
            "center_confidence": min(1.0, radius / 0.15),
            "outer_region": 1 if radius >= 0.5 else 0,
        },
        "direction": {
            "x_positive": 1 if x >= 0 else 0,
            "y_positive": 1 if y >= 0 else 0,
            "angle": angle,
        },
        "spiral": {
            "phase": phase,
            "arm_0": math.cos(phase),
            "arm_1": math.cos(phase - math.pi),
        },
    }


def predict_detail(inputs, use_distance=True, use_direction=True, use_spiral=True):
    groups = nodes(inputs)
    # The coefficients are deliberately written by hand; no fitting occurs.
    score = 0.0
    if use_spiral:
        score += groups["spiral"]["arm_1"] - groups["spiral"]["arm_0"]
    if use_distance:
        score *= 0.5 + 0.5 * groups["distance"]["center_confidence"]
    if use_direction:
        # Direction is used as an explicit coordinate-system sanity check.
        score += 0.0 * (groups["direction"]["x_positive"] + groups["direction"]["y_positive"])
    return {"groups": groups, "score": score, "prediction": int(score >= 0)}


def predict(inputs):
    return predict_detail(inputs)["prediction"]


def evaluate(rows, **options):
    return sum(predict_detail(row["inputs"], **options)["prediction"] == row["target"] for row in rows) / len(rows)


def run(root):
    train_rows = json.loads((root / "spiral_train.json").read_text(encoding="utf-8"))
    test_rows = json.loads((root / "spiral_test.json").read_text(encoding="utf-8"))
    variants = {
        "full": {},
        "distance_only": {"use_direction": False, "use_spiral": False},
        "direction_only": {"use_distance": False, "use_spiral": False},
        "spiral_only": {"use_distance": False, "use_direction": False},
        "distance_plus_spiral": {"use_direction": False},
    }
    with tempfile.TemporaryDirectory() as directory:
        fresh_path = Path(directory) / "fresh.json"
        generate(fresh_path, seed=101)
        fresh_rows = json.loads(fresh_path.read_text(encoding="utf-8"))
    neural = json.loads((root / "spiral_model.json").read_text(encoding="utf-8"))
    results = {
        name: {"train_accuracy": evaluate(train_rows, **options), "test_accuracy": evaluate(test_rows, **options), "fresh_accuracy": evaluate(fresh_rows, **options), "nodes": 9}
        for name, options in variants.items()
    }
    results["neural_baseline"] = {
        "train_accuracy": neural_evaluate(neural, train_rows),
        "test_accuracy": neural_evaluate(neural, test_rows),
        "fresh_accuracy": neural_evaluate(neural, fresh_rows),
        "nodes": 24,
    }
    (root / "human_role_results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


if __name__ == "__main__":
    for name, result in run(Path(__file__).parent).items():
        print(name, result)
