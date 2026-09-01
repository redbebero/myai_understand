"""Compare early cross-entropy training with the existing direct-overlap inverse."""

import json
from pathlib import Path

import numpy as np

from .direct_overlap_inverse_experiment import _direct_constraints, _direct_metrics
from .distribution_inverse_experiment import _pair_distribution
from .generalization_experiment import _adam_update, _copy, _forward, _gradients, _init_model
from .inverse_geometry_experiment import _apply_hidden_delta, _gate_state, _hidden_vector, _inverse_delta
from .margin_inverse_experiment import _geometry, _margin_metrics
from .validation_selective_inverse_experiment import _strict_split


UPDATES = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.001


def _loss(model, inputs, targets):
    probabilities = _forward(model, inputs, 2)[2]
    return float(-np.log(np.maximum(probabilities[np.arange(len(targets)), targets], 1e-12)).mean())


def _pair_geometry(model, inputs, targets):
    h2 = _forward(model, inputs, 2)[0][-1]
    means = [h2[targets == label].mean(axis=0) for label in range(6)]
    direction = means[3] - means[4]
    return {"separation_ratio": _geometry(model, inputs, targets)["separation_ratio"], "pair_centroid_distance": float(np.linalg.norm(direction)), "pair_direction": direction.tolist(), "pair_distribution": {**_direct_metrics(model, inputs, targets), **_pair_distribution(model, inputs, targets)}}


def _cosine(first, second):
    return float(np.dot(first, second) / max(np.linalg.norm(first) * np.linalg.norm(second), 1e-12))


def _geometry_similarity(validation, test, initial):
    val_direction, test_direction = np.asarray(validation["pair_direction"]), np.asarray(test["pair_direction"])
    values_val = np.array([validation["pair_centroid_distance"], validation["pair_distribution"]["wrong_probability_mass"]])
    values_test = np.array([test["pair_centroid_distance"], test["pair_distribution"]["wrong_probability_mass"]])
    initial_scale = np.linalg.norm(np.array([initial["pair_centroid_distance"], initial["pair_distribution"]["wrong_probability_mass"]]))
    return {"pair_direction_cosine": _cosine(val_direction, test_direction), "metric_gap_normalized": float(np.linalg.norm(values_val - values_test) / max(initial_scale, 1e-12))}


def _class_movement(model, baseline, inputs, targets):
    current_h = _forward(model, inputs, 2)[0][-1]
    base_h = _forward(baseline, inputs, 2)[0][-1]
    movements = []
    for label in range(6):
        movements.append((current_h[targets == label] - base_h[targets == label]).mean(axis=0))
    pairwise = [_cosine(movements[i], movements[j]) for i in range(6) for j in range(i + 1, 6) if np.linalg.norm(movements[i]) > 1e-12 and np.linalg.norm(movements[j]) > 1e-12]
    return {"mean_movement_norm": float(np.mean([np.linalg.norm(value) for value in movements])), "class_direction_cosine_mean": float(np.mean(pairwise)) if pairwise else 0.0, "class_movement_norms": [float(np.linalg.norm(value)) for value in movements]}


def _record(model, baseline, data, seen_train, update, method):
    train_geometry = _pair_geometry(model, data["train_x"], data["train_y"])
    val_geometry = _pair_geometry(model, data["val_x"], data["val_y"])
    test_geometry = _pair_geometry(model, data["test_x"], data["test_y"])
    initial_geometry = _pair_geometry(baseline, data["val_x"], data["val_y"])
    return {"method": method, "update": update, "loss": {"train": _loss(model, data["train_x"], data["train_y"]), "validation": _loss(model, data["val_x"], data["val_y"]), "test": _loss(model, data["test_x"], data["test_y"])}, "geometry": {"train": train_geometry, "validation": val_geometry, "test": test_geometry, "validation_test_similarity": _geometry_similarity(val_geometry, test_geometry, initial_geometry)}, "hidden_movement": _class_movement(model, baseline, data["val_x"], data["val_y"]), "coverage_fraction": float(len(seen_train) / len(data["train_y"])) if method == "gradient" else 1.0}


def _run_seed(data, seed):
    initial = _init_model(561, (64, 32), 6, seed)
    baseline = _copy(initial)
    gradient_model = _copy(initial)
    inverse_model = _copy(initial)
    moments = {name: [np.zeros_like(value), np.zeros_like(value)] for name, value in gradient_model.items()}
    rng = np.random.default_rng(seed + 1)
    batches = np.array_split(rng.permutation(len(data["train_y"])), max(1, len(data["train_y"]) // BATCH_SIZE))[:UPDATES]
    seen = set()
    gradient_records = [_record(gradient_model, baseline, data, seen, 0, "gradient")]
    inverse_records = [_record(inverse_model, baseline, data, set(), 0, "inverse")]
    for update, indices in enumerate(batches, 1):
        seen.update(indices.tolist())
        _, _, _, gradients = _gradients(gradient_model, data["train_x"][indices], data["train_y"][indices], 2)
        _adam_update(gradient_model, gradients, moments, update, LEARNING_RATE)
        gradient_records.append(_record(gradient_model, baseline, data, seen, update, "gradient"))
        jacobian, target, _ = _direct_constraints(inverse_model, data["val_x"], data["val_y"])
        inverse_step = _inverse_delta(jacobian, target)
        inverse_model = _apply_hidden_delta(inverse_model, inverse_step, fraction=0.2)
        inverse_records.append(_record(inverse_model, baseline, data, set(), update, "inverse"))
    return {"seed": seed, "gradient": gradient_records, "inverse": inverse_records}


def _trajectory_summary(runs, method):
    return [{"update": update, "train_loss": float(np.mean([run[method][update]["loss"]["train"] for run in runs])), "validation_loss": float(np.mean([run[method][update]["loss"]["validation"] for run in runs])), "test_loss": float(np.mean([run[method][update]["loss"]["test"] for run in runs])), "validation_separation": float(np.mean([run[method][update]["geometry"]["validation"]["separation_ratio"] for run in runs])), "test_separation": float(np.mean([run[method][update]["geometry"]["test"]["separation_ratio"] for run in runs])), "validation_overlap": float(np.mean([run[method][update]["geometry"]["validation"]["pair_distribution"]["wrong_probability_mass"] for run in runs])), "test_overlap": float(np.mean([run[method][update]["geometry"]["test"]["pair_distribution"]["wrong_probability_mass"] for run in runs])), "validation_boundary_overlap": float(np.mean([run[method][update]["geometry"]["validation"]["pair_distribution"]["boundary_overlap"] for run in runs])), "test_boundary_overlap": float(np.mean([run[method][update]["geometry"]["test"]["pair_distribution"]["boundary_overlap"] for run in runs])), "validation_boundary_variance": float(np.mean([np.mean([run[method][update]["geometry"]["validation"]["pair_distribution"][str(label)]["variance"] for label in (3, 4)]) for run in runs])), "test_boundary_variance": float(np.mean([np.mean([run[method][update]["geometry"]["test"]["pair_distribution"][str(label)]["variance"] for label in (3, 4)]) for run in runs])), "hidden_movement": float(np.mean([run[method][update]["hidden_movement"]["mean_movement_norm"] for run in runs])), "class_direction_cosine": float(np.mean([run[method][update]["hidden_movement"]["class_direction_cosine_mean"] for run in runs])), "val_test_direction_cosine": float(np.mean([run[method][update]["geometry"]["validation_test_similarity"]["pair_direction_cosine"] for run in runs])), "val_test_metric_gap": float(np.mean([run[method][update]["geometry"]["validation_test_similarity"]["metric_gap_normalized"] for run in runs])), "coverage_fraction": float(np.mean([run[method][update]["coverage_fraction"] for run in runs]))} for update in range(UPDATES + 1)]


def run_gradient_vs_inverse(data_dir, seeds=(7, 11, 19, 23, 31)):
    data = _strict_split(data_dir)
    runs = [_run_seed(data, seed) for seed in seeds]
    return {"settings": {"architecture": "561→64→32→6", "seeds": list(seeds), "updates": UPDATES, "batch_size": BATCH_SIZE, "split": "strict train/validation/test", "gradient": "Adam cross-entropy on train split", "inverse": "existing direct-overlap inverse on validation split"}, "summary": {"gradient": _trajectory_summary(runs, "gradient"), "inverse": _trajectory_summary(runs, "inverse")}, "runs": runs}


def write_report(result, path):
    lines = ["# Gradient training vs geometry inverse", "", "동일한 random initialization에서 Adam cross-entropy training은 train split mini-batch를 사용하고, inverse는 validation 전체의 기존 direct-overlap objective를 사용했다. test는 update에 사용하지 않고 trajectory 기록용 평가만 수행했다.", "", "## Update 0 → 10"]
    for method in ("gradient", "inverse"):
        start, end = result["summary"][method][0], result["summary"][method][-1]
        lines += [f"### {method}", f"- train loss: {start['train_loss']:.3f} → {end['train_loss']:.3f}", f"- validation loss: {start['validation_loss']:.3f} → {end['validation_loss']:.3f}", f"- test loss: {start['test_loss']:.3f} → {end['test_loss']:.3f}", f"- validation overlap: {start['validation_overlap']:.3f} → {end['validation_overlap']:.3f}", f"- test overlap: {start['test_overlap']:.3f} → {end['test_overlap']:.3f}", f"- test boundary overlap: {start['test_boundary_overlap']:.3f} → {end['test_boundary_overlap']:.3f}", f"- test boundary variance: {start['test_boundary_variance']:.3f} → {end['test_boundary_variance']:.3f}", f"- validation/test direction cosine at end: {end['val_test_direction_cosine']:.3f}", f"- class movement cosine at end: {end['class_direction_cosine']:.3f}", f"- update-data coverage at end: {end['coverage_fraction']:.3f}", ""]
    lines += ["## 최소 원리", "", "gradient training은 각 batch의 cross-entropy를 통해 output layer와 모든 hidden parameter를 동시에 업데이트하고, 서로 다른 class/sample의 gradient가 반복적으로 합쳐진다. inverse는 validation objective와 선택한 보존 제약을 만족하는 국소 해를 최소 norm으로 찾으므로 목표에는 직접적이지만, unseen distribution의 covariance와 모든 class loss를 동시에 학습한 것은 아니다.", "", "`random representation → data-averaged gradient updates → shared representation geometry → validation/test alignment → generalization`", "", "상세 update trajectory와 seed별 결과는 JSON에 저장했다."]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    root = Path(__file__).parent
    result = run_gradient_vs_inverse(root / "UCI HAR Dataset")
    (root / "gradient_vs_inverse_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_report(result, root / "gradient_vs_inverse_analysis.md")
    print(json.dumps({"runs": len(result["runs"]), "result": str(root / "gradient_vs_inverse_results.json")}, indent=2))
