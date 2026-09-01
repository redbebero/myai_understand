"""Train and benchmark the original MLP against a real 8D bottleneck."""

import json
import time
from pathlib import Path

import numpy as np

from .generalization_experiment import _adam_update, _forward, _gradients, _init_model
from .validation_selective_inverse_experiment import _strict_split


SEEDS = (7, 11, 19, 23, 31)
EPOCHS = 80
BATCH_SIZE = 128
LEARNING_RATE = 0.001
CLASSES = 6
CONFIGS = {"original_32d": (64, 32), "bottleneck_8d": (64, 8)}


def count_parameters(input_size, hidden, classes):
    sizes = (input_size,) + tuple(hidden) + (classes,)
    return int(sum(left * right + right for left, right in zip(sizes, sizes[1:])))


def activation_bytes(hidden, classes, bytes_per_value=4):
    return int((sum(hidden) + classes) * bytes_per_value)


def operation_counts(input_size, hidden, classes):
    sizes = (input_size,) + tuple(hidden) + (classes,)
    macs = int(sum(left * right for left, right in zip(sizes, sizes[1:])))
    return macs, 2 * macs


def _train_model(data, hidden, seed):
    model = _init_model(data["train_x"].shape[1], hidden, CLASSES, seed)
    moments = {name: [np.zeros_like(value), np.zeros_like(value)] for name, value in model.items()}
    rng = np.random.default_rng(seed + 1)
    step = 0
    for _ in range(EPOCHS):
        for indices in np.array_split(rng.permutation(len(data["train_y"])), max(1, len(data["train_y"]) // BATCH_SIZE)):
            gradients = _gradients(model, data["train_x"][indices], data["train_y"][indices], len(hidden))[3]
            step += 1
            _adam_update(model, gradients, moments, step, LEARNING_RATE)
    return model


def _metrics(model, inputs, targets, hidden_count):
    probabilities = _forward(model, inputs, hidden_count)[2]
    target_probability = np.maximum(probabilities[np.arange(len(targets)), targets], 1e-12)
    return {"accuracy": float(np.mean(probabilities.argmax(axis=1) == targets)), "cross_entropy": float(-np.log(target_probability).mean())}


def _benchmark(model, inputs, targets, hidden_count, repeats=5):
    _metrics(model, inputs[:1], targets[:1], hidden_count)
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        _metrics(model, inputs, targets, hidden_count)
        durations.append(time.perf_counter() - start)
    return {"median_full_test_ms": float(np.median(durations) * 1000), "repeats": repeats}


def run_bottleneck_experiment(data_dir, seeds=SEEDS):
    data = _strict_split(data_dir)
    runs = {name: [] for name in CONFIGS}
    for seed in seeds:
        for name, hidden in CONFIGS.items():
            model = _train_model(data, hidden, seed)
            metrics = _metrics(model, data["test_x"], data["test_y"], len(hidden))
            benchmark = _benchmark(model, data["test_x"], data["test_y"], len(hidden))
            macs, flops = operation_counts(data["train_x"].shape[1], hidden, CLASSES)
            runs[name].append({"seed": seed, **metrics, **benchmark, "hidden": list(hidden), "parameter_count": count_parameters(data["train_x"].shape[1], hidden, CLASSES), "parameter_storage_bytes_fp32": count_parameters(data["train_x"].shape[1], hidden, CLASSES) * 4, "activation_bytes_fp32": activation_bytes(hidden, CLASSES), "macs": macs, "flops": flops})
    summary = {}
    for name, values in runs.items():
        summary[name] = {key: float(np.mean([row[key] for row in values])) for key in ("accuracy", "cross_entropy", "median_full_test_ms", "parameter_count", "parameter_storage_bytes_fp32", "activation_bytes_fp32", "macs", "flops")}
    return {"settings": {"architecture_original": "561→64→32→6", "architecture_bottleneck": "561→64→8→6", "seeds": list(seeds), "epochs": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE, "optimizer": "Adam", "split": "strict train/validation/test", "activation_definition": "all non-input layer outputs, FP32, one sample", "flops_definition": "2 FLOPs per multiply-accumulate; ReLU/softmax excluded"}, "summary": summary, "runs": runs}


def write_report(result, path):
    original = result["summary"]["original_32d"]
    bottleneck = result["summary"]["bottleneck_8d"]
    pct = lambda value: (value / original["accuracy"] - 1) * 100
    lines = ["# 실제 8차원 bottleneck 비교", "", "동일한 조건으로 원본 `561→64→32→6`과 bottleneck `561→64→8→6`을 각각 처음부터 학습했다. 이전 post-hoc projection과 달리 8차원 hidden layer 자체가 학습된다.", "", "| 지표 | 원본 32D | bottleneck 8D | 변화율 |", "|---|---:|---:|---:|", f"| Test accuracy | {original['accuracy']:.3f} | {bottleneck['accuracy']:.3f} | {pct(bottleneck['accuracy']):+.2f}% |", f"| Cross-entropy | {original['cross_entropy']:.3f} | {bottleneck['cross_entropy']:.3f} | {(bottleneck['cross_entropy'] / original['cross_entropy'] - 1) * 100:+.2f}% |", f"| Parameters | {original['parameter_count']:.0f} | {bottleneck['parameter_count']:.0f} | {(bottleneck['parameter_count'] / original['parameter_count'] - 1) * 100:+.2f}% |", f"| FP32 model storage | {original['parameter_storage_bytes_fp32'] / 1024:.1f} KB | {bottleneck['parameter_storage_bytes_fp32'] / 1024:.1f} KB | {(bottleneck['parameter_storage_bytes_fp32'] / original['parameter_storage_bytes_fp32'] - 1) * 100:+.2f}% |", f"| FP32 activation memory/sample | {original['activation_bytes_fp32']:.0f} B | {bottleneck['activation_bytes_fp32']:.0f} B | {(bottleneck['activation_bytes_fp32'] / original['activation_bytes_fp32'] - 1) * 100:+.2f}% |", f"| Median inference time | {original['median_full_test_ms']:.2f} ms | {bottleneck['median_full_test_ms']:.2f} ms | {(bottleneck['median_full_test_ms'] / original['median_full_test_ms'] - 1) * 100:+.2f}% |", f"| MACs/sample | {original['macs']:.0f} | {bottleneck['macs']:.0f} | {(bottleneck['macs'] / original['macs'] - 1) * 100:+.2f}% |", f"| FLOPs/sample | {original['flops']:.0f} | {bottleneck['flops']:.0f} | {(bottleneck['flops'] / original['flops'] - 1) * 100:+.2f}% |", "", "## 해석", "", "이 결과는 post-hoc 표현 압축이 아니라 실제로 8차원 hidden layer를 학습한 결과다. 파라미터와 연산량 감소 폭은 첫 번째 561→64 층이 대부분의 비용을 차지하기 때문에 제한적이다. activation memory는 hidden 32차원에서 8차원으로 줄어든 만큼 감소한다. 추론 시간은 CPU·BLAS 환경의 측정값이므로 방향성 참고용이다."]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    root = Path(__file__).parent
    result = run_bottleneck_experiment(root / "UCI HAR Dataset")
    (root / "bottleneck_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_report(result, root / "bottleneck_analysis.md")
    print(json.dumps(result["summary"], indent=2))
