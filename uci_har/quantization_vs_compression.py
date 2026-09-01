"""Compare ordinary weight quantization with post-hoc representation compression."""

import json
from pathlib import Path

import numpy as np

from .representation_compression_experiment import DIMENSIONS
from .uci_har_experiment import baseline_forward, load_data, train_baseline
from .validation_selective_inverse_experiment import _strict_split


SEEDS = (7, 11, 19, 23, 31)
BITS = (8, 4, 2)


def _quantize_dequantize(values, bits):
    limit = (1 << (bits - 1)) - 1
    scale = np.max(np.abs(values)) / limit if np.max(np.abs(values)) else 1.0
    return np.round(values / scale).clip(-limit - 1, limit) * scale


def _quantized_model(model, bits):
    return {name: _quantize_dequantize(value, bits) for name, value in model.items()}


def model_storage_bytes(model, bits):
    return int(sum(value.size for value in model.values()) * bits / 8 + (1 if bits < 8 and sum(value.size for value in model.values()) * bits % 8 else 0))


def _accuracy(model, inputs, targets):
    return float(np.mean(baseline_forward(model, inputs)[2].argmax(axis=1) == targets))


def _parameter_count(model):
    return int(sum(value.size for value in model.values()))


def run_quantization_comparison(data_dir, seeds=SEEDS):
    data = _strict_split(data_dir)
    runs = []
    for seed in seeds:
        model = train_baseline(data["train_x"], data["train_y"], seed=seed)
        original_accuracy = _accuracy(model, data["test_x"], data["test_y"])
        quantized = {}
        for bits in BITS:
            candidate = _quantized_model(model, bits)
            quantized[str(bits)] = {"accuracy": _accuracy(candidate, data["test_x"], data["test_y"]), "storage_bytes": model_storage_bytes(model, bits)}
        runs.append({"seed": seed, "parameter_count": _parameter_count(model), "original_accuracy": original_accuracy, "original_storage_bytes": model_storage_bytes(model, 32), "quantized": quantized})
    parameter_count = runs[0]["parameter_count"]
    quantized_summary = {str(bits): {"accuracy": float(np.mean([run["quantized"][str(bits)]["accuracy"] for run in runs])), "storage_bytes": model_storage_bytes({"parameters": np.zeros(parameter_count)}, bits)} for bits in BITS}
    original_accuracy = float(np.mean([run["original_accuracy"] for run in runs]))
    original_bytes = model_storage_bytes({"parameters": np.zeros(parameter_count)}, 32)
    representation = {
        str(k): {"accuracy": None, "activation_storage_bytes": k * 4, "model_storage_bytes": original_bytes + (32 * k + 32) * 4, "extra_projection_macs": 2 * 32 * k}
        for k in DIMENSIONS
    }
    return {"settings": {"architecture": "561→64→32→6", "seeds": list(seeds), "quantization": "symmetric per-array dequantized evaluation", "representation": "post-hoc hidden projection and reconstruction", "baseline_parameter_count": parameter_count, "dense_macs": 561 * 64 + 64 * 32 + 32 * 6}, "summary": {"original": {"accuracy": original_accuracy, "model_storage_bytes": original_bytes, "activation_storage_bytes": 32 * 4}, "quantization": quantized_summary, "representation_compression": representation}, "runs": runs}


def write_report(result, path, representation_result_path):
    original = result["summary"]["original"]
    lines = ["# 양자화와 hidden 표현 압축 비교", "", "양자화는 같은 MLP 가중치를 낮은 bit로 저장한 뒤 복원해 평가했다. 표현 압축은 기존 실험 결과의 frozen output-layer 방식이며, 원래 MLP를 제거하지 않고 hidden 표현만 투영·복원한다.", "", "| 방식 | 정확도 | 저장 크기 | hidden 활성값 |", "|---|---:|---:|---:|"]
    lines.append(f"| FP32 원본 | {original['accuracy']:.3f} | {original['model_storage_bytes'] / 1024:.1f} KB | {original['activation_storage_bytes']} B |")
    for bits, row in result["summary"]["quantization"].items():
        lines.append(f"| INT{bits} 양자화 | {row['accuracy']:.3f} | {row['storage_bytes'] / 1024:.1f} KB | 128 B |")
    compression = json.loads(Path(representation_result_path).read_text())["summary"]
    for method in ("pca", "class_separating", "supervised_output"):
        row = compression[method]["8"]
        info = result["summary"]["representation_compression"]["8"]
        lines.append(f"| {method} k=8 | {row['accuracy']:.3f} | {info['model_storage_bytes'] / 1024:.1f} KB* | {info['activation_storage_bytes']} B |")
    lines += ["", "* 현재 구현은 원래 MLP를 유지하고 projection basis와 평균을 추가하므로 모델 저장 크기와 원래 dense 연산을 줄이지 않는다. 줄어드는 것은 hidden activation 저장량이다.", "", f"원래 dense MACs: {result['settings']['dense_macs']:,}. k=8 projection/reconstruction 추가 MACs: {result['summary']['representation_compression']['8']['extra_projection_macs']:,}.", "", "## 결론", "", "같은 정확도에 가까운 조건에서 INT8은 모델 저장 크기를 약 4분의 1로 줄이는 반면, 현재의 표현 압축은 k=8 hidden activation을 4분의 1로 줄이지만 모델 자체는 작아지지 않는다. 실제 모델 크기와 연산량까지 줄이려면 projection을 학습 그래프 안에 넣고 앞뒤 weight를 구조적으로 다시 접어야 한다."]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    root = Path(__file__).parent
    result = run_quantization_comparison(root / "UCI HAR Dataset")
    (root / "quantization_vs_compression_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_report(result, root / "quantization_vs_compression_analysis.md", root / "representation_compression_results.json")
    print(json.dumps(result["summary"], indent=2))
