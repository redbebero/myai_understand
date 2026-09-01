"""Compare raw-CNN behavior after weight-only low-bit quantization."""

import json
import time
from pathlib import Path

import numpy as np

from .raw_cnn_experiment import accuracy, load_raw, quantized_model


def benchmark(model, inputs, targets, repeats=3):
    scores, durations = [], []
    for _ in range(repeats):
        start = time.perf_counter()
        scores.append(accuracy(model, inputs, targets))
        durations.append(time.perf_counter() - start)
    return {"accuracy": float(np.mean(scores)), "milliseconds": float(np.median(durations) * 1000)}


def run():
    root = Path(__file__).parent
    data = load_raw(root.parent / "data" / "UCI HAR Dataset")
    baseline = dict(np.load(root / "baseline_model.npz"))
    results = {"original": benchmark(baseline, data["test_x"], data["test_y"])}
    for bits in (8, 4, 2):
        results[f"{bits}_bit"] = benchmark(quantized_model(baseline, bits), data["test_x"], data["test_y"])
    results["parameter_count"] = sum(value.size for value in baseline.values())
    results["weight_storage_bytes_theoretical"] = {str(bits): int(sum(value.size for value in baseline.values()) * bits / 8) for bits in (8, 4, 2)}
    (root / "quantization_results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
