"""Validation-only selective inverse design with safe-sample constraints."""

import json
from pathlib import Path

import numpy as np

from .generalization_experiment import _copy, _forward, _gradients, _init_model
from .inverse_geometry_experiment import (
    ITERATIONS,
    TRAIN_EPOCHS,
    _adam_update,
    _apply_hidden_delta,
    _gate_state,
    _hidden_vector,
    _inverse_delta,
    _same_norm_delta,
    _train_model,
)
from .margin_inverse_experiment import (
    _design_jacobian,
    _geometry,
    _hidden_rows,
    _margin_metrics,
    _margin_state,
    _margin_target,
)
from .uci_har_experiment import _read_matrix


CLASS_COUNT = 6
VAL_FRACTION = 0.2
SPLIT_SEED = 2026
VULNERABLE_MARGIN = 1.0
MAX_VULNERABLE_PER_CLASS = 20
SAFE_ANCHORS_PER_CLASS = 10
TARGET_MARGIN = 1.0
GAIN_CAP = 2.0
STEP_FRACTION = 0.2


def _strict_split(data_dir):
    data_dir = Path(data_dir)
    x = _read_matrix(data_dir / "train" / "X_train.txt")
    y = _read_matrix(data_dir / "train" / "y_train.txt").astype(int).ravel() - 1
    test_x = _read_matrix(data_dir / "test" / "X_test.txt")
    test_y = _read_matrix(data_dir / "test" / "y_test.txt").astype(int).ravel() - 1
    rng = np.random.default_rng(SPLIT_SEED)
    train_indices, val_indices = [], []
    for label in range(CLASS_COUNT):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        cut = int(len(indices) * (1.0 - VAL_FRACTION))
        train_indices.extend(indices[:cut])
        val_indices.extend(indices[cut:])
    train_indices, val_indices = np.asarray(train_indices), np.asarray(val_indices)
    mean = x[train_indices].mean(axis=0)
    scale = x[train_indices].std(axis=0)
    scale[scale == 0] = 1.0
    transform = lambda values: (values - mean) / scale
    return {"train_x": transform(x[train_indices]), "train_y": y[train_indices], "val_x": transform(x[val_indices]), "val_y": y[val_indices], "test_x": transform(test_x), "test_y": test_y}


def _partition(model, inputs, targets):
    margins, _, _, _ = _margin_state(model, inputs, targets)
    misclassified = margins < 0.0
    vulnerable = margins < VULNERABLE_MARGIN
    safe = margins >= VULNERABLE_MARGIN
    return {"margins": margins, "misclassified": np.flatnonzero(misclassified), "vulnerable": np.flatnonzero(vulnerable), "safe": np.flatnonzero(safe)}


def _choose(indices, margins, targets, per_class, reverse=False):
    selected = []
    for label in range(CLASS_COUNT):
        candidates = indices[targets[indices] == label]
        ordered = candidates[np.argsort(margins[candidates])]
        if reverse:
            ordered = ordered[::-1]
        selected.extend(ordered[:per_class])
    return np.asarray(selected, dtype=int)


def _selective_constraints(model, inputs, targets, baseline_partition):
    margins, wrong, _, _ = _margin_state(model, inputs, targets)
    vulnerable = _choose(baseline_partition["vulnerable"], baseline_partition["margins"], targets, MAX_VULNERABLE_PER_CLASS)
    safe = _choose(baseline_partition["safe"], baseline_partition["margins"], targets, SAFE_ANCHORS_PER_CLASS, reverse=True)
    selected = np.concatenate([vulnerable, safe])
    directions, gains, kinds = [], [], []
    for index in vulnerable:
        label, competitor = int(targets[index]), int(wrong[index])
        direction = model["w2"][:, label] - model["w2"][:, competitor]
        directions.append(direction)
        gains.append(float(np.clip(TARGET_MARGIN - margins[index], 0.0, GAIN_CAP)))
        kinds.append("vulnerable")
    for index in safe:
        label, competitor = int(targets[index]), int(wrong[index])
        directions.append(model["w2"][:, label] - model["w2"][:, competitor])
        gains.append(0.0)
        kinds.append("safe_anchor")
    rows = _hidden_rows(model, inputs[selected]).reshape(len(selected), 32, -1)
    jacobian = np.einsum("ij,ijp->ip", np.asarray(directions), rows)
    return selected, np.asarray(directions), np.asarray(gains), kinds, jacobian


def _full_margin_inverse(model, inputs, targets):
    selected, directions, gains, _ = _margin_target(model, inputs, targets)
    return _inverse_delta(_design_jacobian(model, inputs, targets, selected, directions), gains)


def _classification(model, inputs, targets):
    return _margin_metrics(model, inputs, targets)


def _safe_preservation(baseline_model, candidate, inputs, targets, safe_indices):
    base_margin, _, base_h2, _ = _margin_state(baseline_model, inputs[safe_indices], targets[safe_indices])
    new_margin, _, new_h2, _ = _margin_state(candidate, inputs[safe_indices], targets[safe_indices])
    return {"mean_abs_margin_change": float(np.mean(np.abs(new_margin - base_margin))), "mean_hidden_movement": float(np.mean(np.linalg.norm(new_h2 - base_h2, axis=1))), "margin_preserved_fraction": float(np.mean(np.abs(new_margin - base_margin) < 0.1))}


def _evaluate_candidate(baseline_model, candidate, data, baseline_test_safe):
    test_metrics = _classification(candidate, data["test_x"], data["test_y"])
    safe_targets = data["test_y"][baseline_test_safe]
    safe_predictions = _forward(candidate, data["test_x"][baseline_test_safe], 2)[2].argmax(axis=1)
    test_metrics["safe_accuracy_preservation"] = float(np.mean(safe_predictions == safe_targets))
    return {"test": test_metrics, "geometry": _geometry(candidate, data["test_x"], data["test_y"]), "safe_test_count": int(np.sum(baseline_test_safe))}


def run_seed(data, seed):
    model = _train_model(_init_model(561, (64, 32), CLASS_COUNT, seed), data["train_x"], data["train_y"], seed)
    validation_partition = _partition(model, data["val_x"], data["val_y"])
    validation_baseline = _classification(model, data["val_x"], data["val_y"])
    selective = _selective_constraints(model, data["val_x"], data["val_y"], validation_partition)
    selective_delta = _inverse_delta(selective[-1], selective[2])
    full_delta = _full_margin_inverse(model, data["val_x"], data["val_y"])
    gradient = _gradients(model, data["train_x"], data["train_y"], 2)[3]
    gradient_vector = np.concatenate([gradient["w0"].ravel(), gradient["b0"], gradient["w1"].ravel(), gradient["b1"]])
    rng = np.random.default_rng(seed + 9200)
    norm = np.linalg.norm(selective_delta)
    candidates = {"selective_inverse": _apply_hidden_delta(model, selective_delta), "full_margin_inverse": _apply_hidden_delta(model, full_delta), "random_same_norm": _apply_hidden_delta(model, _same_norm_delta(selective_delta, rng)), "gradient_same_norm": _apply_hidden_delta(model, -gradient_vector * (norm / max(np.linalg.norm(gradient_vector), 1e-12)))}
    candidate_results = {}
    for name, candidate in candidates.items():
        candidate_results[name] = {"model": candidate, "validation": _classification(candidate, data["val_x"], data["val_y"]), "safe_validation": _safe_preservation(model, candidate, data["val_x"], data["val_y"], validation_partition["safe"]), "delta_norm": float(np.linalg.norm(_hidden_vector(candidate) - _hidden_vector(model))), "gate_change_fraction": float(np.mean(_gate_state(candidate, data["val_x"]) != _gate_state(model, data["val_x"]))) }
    current = _copy(model)
    iterative = []
    for iteration in range(ITERATIONS):
        _, _, gains, _, jacobian = _selective_constraints(current, data["val_x"], data["val_y"], validation_partition)
        step = _inverse_delta(jacobian, gains)
        current = _apply_hidden_delta(current, step, STEP_FRACTION)
        iterative.append({"model": _copy(current), "validation": _classification(current, data["val_x"], data["val_y"]), "safe_validation": _safe_preservation(model, current, data["val_x"], data["val_y"], validation_partition["safe"]), "iteration": iteration + 1, "step_norm": float(np.linalg.norm(step) * STEP_FRACTION)})
    baseline_test_margin, _, _, _ = _margin_state(model, data["test_x"], data["test_y"])
    baseline_test_safe = baseline_test_margin >= VULNERABLE_MARGIN
    baseline_test = _evaluate_candidate(model, model, data, baseline_test_safe)
    for name, result in candidate_results.items():
        candidate = result.pop("model")
        result.update(_evaluate_candidate(model, candidate, data, baseline_test_safe))
    for result in iterative:
        candidate = result.pop("model")
        result.update(_evaluate_candidate(model, candidate, data, baseline_test_safe))
    return {"seed": seed, "validation_partition": {key: int(len(value)) for key, value in validation_partition.items() if key != "margins"}, "validation_baseline": validation_baseline, "baseline_test": baseline_test, "jacobian_shape": list(selective[-1].shape), "one_shot": candidate_results, "iterative": iterative}


def run_validation_selective(data_dir, seeds=(7, 11, 19, 23, 31)):
    data = _strict_split(data_dir)
    runs = [run_seed(data, seed) for seed in seeds]
    names = ("selective_inverse", "full_margin_inverse", "random_same_norm", "gradient_same_norm")
    summary = {"validation_partitions": {key: float(np.mean([run["validation_partition"][key] for run in runs])) for key in ("misclassified", "vulnerable", "safe")}, "one_shot": {name: {"accuracy": float(np.mean([r["one_shot"][name]["test"]["accuracy"] for r in runs])), "misclassified": float(np.mean([sum(sum(row) for row in r["one_shot"][name]["test"]["confusion"]) - sum(r["one_shot"][name]["test"]["confusion"][i][i] for i in range(CLASS_COUNT)) for r in runs])), "q10_margin": float(np.mean([r["one_shot"][name]["test"]["q10_margin"] for r in runs])), "min_margin": float(np.mean([r["one_shot"][name]["test"]["min_margin"] for r in runs])), "safe_accuracy_preservation": float(np.mean([r["one_shot"][name]["test"]["safe_accuracy_preservation"] for r in runs])), "safe_margin_change": float(np.mean([r["one_shot"][name]["safe_validation"]["mean_abs_margin_change"] for r in runs])), "gate_change_fraction": float(np.mean([r["one_shot"][name]["gate_change_fraction"] for r in runs]))} for name in names}, "baseline": {"accuracy": float(np.mean([r["baseline_test"]["test"]["accuracy"] for r in runs])), "q10_margin": float(np.mean([r["baseline_test"]["test"]["q10_margin"] for r in runs])), "min_margin": float(np.mean([r["baseline_test"]["test"]["min_margin"] for r in runs]))}, "iterative": {"accuracy": [float(np.mean([r["iterative"][i]["test"]["accuracy"] for r in runs])) for i in range(ITERATIONS)], "q10_margin": [float(np.mean([r["iterative"][i]["test"]["q10_margin"] for r in runs])) for i in range(ITERATIONS)], "safe_accuracy_preservation": [float(np.mean([r["iterative"][i]["test"]["safe_accuracy_preservation"] for r in runs])) for i in range(ITERATIONS)]}}
    return {"settings": {"architecture": "561→64→32→6", "seeds": list(seeds), "split": "train 80% / validation 20% / test untouched until final evaluation", "train_epochs": TRAIN_EPOCHS, "vulnerable_margin": VULNERABLE_MARGIN, "max_vulnerable_per_class": MAX_VULNERABLE_PER_CLASS, "safe_anchors_per_class": SAFE_ANCHORS_PER_CLASS}, "summary": summary, "runs": runs}


def write_report(result, path):
    s = result["summary"]
    validation_baseline = np.mean([run["validation_baseline"]["accuracy"] for run in result["runs"]])
    validation_selective = np.mean([run["one_shot"]["selective_inverse"]["validation"]["accuracy"] for run in result["runs"]])
    lines = ["# Validation-only selective hidden geometry inverse", "", "train의 80%로 모델을 학습하고 validation에서만 취약 sample과 safe anchor를 선택했다. test는 모든 update가 끝난 뒤 최종 평가에만 사용했다.", "", "## Validation partition", "", f"평균 misclassified={s['validation_partitions']['misclassified']:.1f}, vulnerable={s['validation_partitions']['vulnerable']:.1f}, safe={s['validation_partitions']['safe']:.1f}", f"validation accuracy: baseline={validation_baseline:.3f} → selective={validation_selective:.3f}", "", "## Test 결과", f"baseline: accuracy={s['baseline']['accuracy']:.3f}, q10 margin={s['baseline']['q10_margin']:.3f}, min margin={s['baseline']['min_margin']:.3f}"]
    for name, value in s["one_shot"].items():
        lines.append(f"- {name}: accuracy={value['accuracy']:.3f}, misclassified={value['misclassified']:.1f}, q10 margin={value['q10_margin']:.3f}, min margin={value['min_margin']:.3f}, safe accuracy={value['safe_accuracy_preservation']:.3f}, safe validation margin change={value['safe_margin_change']:.3f}, gate change={value['gate_change_fraction']:.1%}")
    lines += ["", "## Repeated selective inverse", f"- accuracy: {', '.join(f'{x:.3f}' for x in s['iterative']['accuracy'])}", f"- q10 margin: {', '.join(f'{x:.3f}' for x in s['iterative']['q10_margin'])}", f"- safe accuracy: {', '.join(f'{x:.3f}' for x in s['iterative']['safe_accuracy_preservation'])}", "", "## 결론", "", "validation에서는 선택 inverse가 개선되지만 test에서는 baseline보다 정확도가 낮고 오분류가 늘었다. 따라서 safe sample 보존 제약은 일반화됐지만, validation sample별 수정 방향은 test distribution 전체의 공통 경계 방향으로 일반화되지 않았다. confusion matrix와 seed별 원자료는 JSON에 저장했다."]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    root = Path(__file__).parent
    result = run_validation_selective(root / "UCI HAR Dataset")
    (root / "validation_selective_inverse_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_report(result, root / "validation_selective_inverse_analysis.md")
    print(json.dumps({"runs": len(result["runs"]), "result": str(root / "validation_selective_inverse_results.json")}, indent=2))
