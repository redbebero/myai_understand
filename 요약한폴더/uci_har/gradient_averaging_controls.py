"""Exposure-matched controls for sample-gradient averaging."""

import json
from pathlib import Path

import numpy as np

from .generalization_experiment import _adam_update, _copy, _init_model
from .gradient_averaging_experiment import (
    BATCH_SIZES,
    _centroid_signature,
    _evaluate,
    _flatten,
    _gradient_decomposition,
    _sample_gradients,
    _unflatten,
)
from .validation_selective_inverse_experiment import _strict_split


EXPOSURE = 1280
FIXED_SET = 128


def _cosine(first, second):
    return float(np.dot(first, second) / max(np.linalg.norm(first) * np.linalg.norm(second), 1e-12))


def _run_sequence(data, seed, sequence, batch_size):
    model = _init_model(561, (64, 32), 6, seed)
    initial = _copy(model)
    names = tuple(model)
    moments = {name: [np.zeros_like(value), np.zeros_like(value)] for name, value in model.items()}
    alignment, cancellation, residual_fraction, sample_count = [], [], [], 0
    update_count = 0
    for start in range(0, len(sequence), batch_size):
        indices = sequence[start:start + batch_size]
        gradients = _sample_gradients(model, data["train_x"][indices], data["train_y"][indices])
        decomposition = _gradient_decomposition(gradients)
        values = _unflatten(decomposition["mean"], model, names)
        _adam_update(model, values, moments, update_count + 1, 0.001)
        count = len(indices)
        alignment.append(decomposition["alignment"] * count)
        cancellation.append(decomposition["residual_cancellation"] * count)
        residual_fraction.append(decomposition["residual_energy_fraction"] * count)
        sample_count += count
        update_count += 1
    delta = _flatten(model, names) - _flatten(initial, names)
    return {"batch_size": batch_size, "updates": update_count, "samples": sample_count, "initial": _evaluate(initial, data), "final": _evaluate(model, data), "gradient_alignment": float(sum(alignment) / max(sample_count, 1)), "residual_cancellation": float(sum(cancellation) / max(sample_count, 1)), "residual_energy_fraction": float(sum(residual_fraction) / max(sample_count, 1)), "parameter_movement_norm": float(np.linalg.norm(delta)), "parameter_movement_direction": delta, "model": model, "initial_model": initial}


def _strip(result):
    return {key: value for key, value in result.items() if key not in ("model", "initial_model", "parameter_movement_direction")}


def _run_seed(data, seed):
    sequence = np.random.default_rng(seed + 1).permutation(len(data["train_y"]))[:EXPOSURE]
    exposure = [_run_sequence(data, seed, sequence, size) for size in BATCH_SIZES]
    fixed_sequence = sequence[:FIXED_SET]
    fixed = [_run_sequence(data, seed, fixed_sequence, size) for size in BATCH_SIZES]
    reference = next(item for item in fixed if item["batch_size"] == 128)["parameter_movement_direction"]
    common_norm = float(np.mean([item["parameter_movement_norm"] for item in fixed]))
    for item in fixed:
        item["movement_direction_cosine_to_batch128"] = _cosine(item["parameter_movement_direction"], reference)
        scale = common_norm / max(item["parameter_movement_norm"], 1e-12)
        candidate = _copy(item["initial_model"])
        for name in candidate:
            candidate[name] += scale * (item["model"][name] - item["initial_model"][name])
        item["same_movement_norm"] = _evaluate(candidate, data)
    return {"seed": seed, "exposure": [_strip(item) for item in exposure], "fixed_set": [_strip(item) for item in fixed]}


def _summary(runs, key):
    result = {}
    for size in BATCH_SIZES:
        items = [next(item for item in run[key] if item["batch_size"] == size) for run in runs]
        final = [item["final"] for item in items]
        result[str(size)] = {"updates": float(np.mean([item["updates"] for item in items])), "samples": float(np.mean([item["samples"] for item in items])), "gradient_alignment": float(np.mean([item["gradient_alignment"] for item in items])), "residual_cancellation": float(np.mean([item["residual_cancellation"] for item in items])), "residual_energy_fraction": float(np.mean([item["residual_energy_fraction"] for item in items])), "parameter_movement_norm": float(np.mean([item["parameter_movement_norm"] for item in items])), "train_loss": float(np.mean([item["final"]["train_loss"] for item in items])), "validation_loss": float(np.mean([item["final"]["validation_loss"] for item in items])), "test_loss": float(np.mean([item["final"]["test_loss"] for item in items])), "validation_accuracy": float(np.mean([item["final"]["validation_accuracy"] for item in items])), "test_accuracy": float(np.mean([item["final"]["test_accuracy"] for item in items])), "test_separation": float(np.mean([item["final"]["test_geometry"]["separation_ratio"] for item in items])), "test_overlap": float(np.mean([item["final"]["test_overlap"] for item in items])), "val_test_geometry_cosine": float(np.mean([item["final"]["geometry_similarity"]["centroid_geometry_cosine"] for item in items]))}
        if key == "fixed_set":
            result[str(size)]["movement_direction_cosine_to_batch128"] = float(np.mean([item["movement_direction_cosine_to_batch128"] for item in items]))
            result[str(size)]["same_norm_test_accuracy"] = float(np.mean([item["same_movement_norm"]["test_accuracy"] for item in items]))
            result[str(size)]["same_norm_test_separation"] = float(np.mean([item["same_movement_norm"]["test_geometry"]["separation_ratio"] for item in items]))
            result[str(size)]["same_norm_val_test_cosine"] = float(np.mean([item["same_movement_norm"]["geometry_similarity"]["centroid_geometry_cosine"] for item in items]))
    return result


def run_controls(data_dir, seeds=(7, 11, 19, 23, 31)):
    data = _strict_split(data_dir)
    runs = [_run_seed(data, seed) for seed in seeds]
    return {"settings": {"architecture": "561→64→32→6", "seeds": list(seeds), "batch_sizes": list(BATCH_SIZES), "optimizer": "Adam cross-entropy", "split": "strict train/validation/test", "exposure": EXPOSURE, "fixed_set": FIXED_SET}, "summary": {"exposure_matched": _summary(runs, "exposure"), "fixed_sample_set": _summary(runs, "fixed_set")}, "runs": runs}


def write_report(result, path):
    lines = ["# Exposure-matched gradient averaging controls", "", "실험 A는 모든 batch size가 같은 1280개 train sample을 보게 했고, 실험 B는 같은 128개 sample을 batch 순서만 다르게 처리했다. test는 update에 사용하지 않았다."]
    for title, key in (("Experiment A: same sample exposure", "exposure_matched"), ("Experiment B: same 128-sample set", "fixed_sample_set")):
        lines += ["", f"## {title}"]
        for size, value in result["summary"][key].items():
            extra = f", same-norm test accuracy={value['same_norm_test_accuracy']:.3f}, same-norm separation={value['same_norm_test_separation']:.3f}" if key == "fixed_sample_set" else ""
            lines.append(f"- batch {size}: updates={value['updates']:.0f}, movement={value['parameter_movement_norm']:.3f}, cancellation={value['residual_cancellation']:.3f}, test accuracy={value['test_accuracy']:.3f}, test separation={value['test_separation']:.3f}, val/test cosine={value['val_test_geometry_cosine']:.3f}{extra}")
    lines += ["", "## 판정", "", "Experiment A에서 batch 차이가 남으면 exposure만으로 설명되지 않는다. Experiment B에서 같은 sample set에서도 동시 averaging이 더 안정적인 geometry를 만들면 averaging 가설을 지지하지만, Adam update 횟수와 parameter movement가 다르므로 그 차이를 함께 해석해야 한다.", "", "`same data → averaging schedule → residual cancellation → representation geometry → unseen generalization`"]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    root = Path(__file__).parent
    result = run_controls(root / "UCI HAR Dataset")
    (root / "gradient_averaging_controls_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_report(result, root / "gradient_averaging_controls_analysis.md")
    print(json.dumps({"seeds": len(result["runs"]), "result": str(root / "gradient_averaging_controls_results.json")}, indent=2))
