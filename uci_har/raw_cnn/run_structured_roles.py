import json
from pathlib import Path

import numpy as np

from .raw_cnn_experiment import load_raw
from .structured_role_model import (
    model_parameter_count,
    save_model,
    structured_accuracy,
    structured_predict,
    train_structured_model,
)


def main():
    root = Path(__file__).parent
    data = load_raw(root.parent / "data" / "UCI HAR Dataset")
    model = train_structured_model(data["train_x"], data["train_y"])
    save_model(root / "structured_role_model.npz", model)
    noisy = data["test_x"] + np.random.default_rng(19).normal(0, 0.05, data["test_x"].shape)
    scaled = data["test_x"] * 1.1
    results = {
        "structure": {"windows": 8, "raw_role_features": 768, "thresholded_event_features": 48, "model_input_features": 816},
        "train": structured_accuracy(model, data["train_x"], data["train_y"]),
        "test": structured_accuracy(model, data["test_x"], data["test_y"]),
        "noise_0.05": structured_accuracy(model, noisy, data["test_y"]),
        "scale_1.1": structured_accuracy(model, scaled, data["test_y"]),
        "parameters": model_parameter_count(model),
        "model_file_bytes": (root / "structured_role_model.npz").stat().st_size,
    }
    results["class_predictions"] = {str(label): int(count) for label, count in zip(*np.unique(structured_predict(model, data["test_x"]), return_counts=True))}
    (root / "structured_role_results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
