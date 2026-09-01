"""Validation-only class-pair subspace inverse design."""

import json
from pathlib import Path

import numpy as np

from .generalization_experiment import _copy, _forward, _gradients, _init_model
from .inverse_geometry_experiment import _apply_hidden_delta, _gate_state, _hidden_vector, _inverse_delta, _same_norm_delta, _train_model
from .margin_inverse_experiment import _design_jacobian, _geometry, _hidden_rows, _margin_metrics, _margin_state, _margin_target
from .validation_selective_inverse_experiment import (
    CLASS_COUNT,
    ITERATIONS,
    SAFE_ANCHORS_PER_CLASS,
    STEP_FRACTION,
    VULNERABLE_MARGIN,
    _classification,
    _partition,
    _safe_preservation,
    _selective_constraints,
    _strict_split,
)


PAIR_GAIN = 0.5


def _confusion_pair(model, inputs, targets):
    confusion = np.asarray(_margin_metrics(model, inputs, targets)["confusion"])
    best, score = (0, 1), -1
    for first in range(CLASS_COUNT):
        for second in range(first + 1, CLASS_COUNT):
            if confusion[first, second] + confusion[second, first] > score:
                best, score = (first, second), int(confusion[first, second] + confusion[second, first])
    return best, confusion.tolist(), score


def _normalize(vector):
    return vector / max(np.linalg.norm(vector), 1e-12)


def _pair_subspace(model, inputs, targets, first, second):
    h2 = _forward(model, inputs, 2)[0][-1]
    means = [h2[targets == label].mean(axis=0) for label in range(CLASS_COUNT)]
    hidden_direction = _normalize(means[first] - means[second])
    decision_direction = _normalize(model["w2"][:, first] - model["w2"][:, second])
    basis, _ = np.linalg.qr(np.column_stack([hidden_direction, decision_direction]))
    rank = int(np.linalg.matrix_rank(np.column_stack([hidden_direction, decision_direction]), tol=1e-8))
    return means, hidden_direction, decision_direction, basis[:, :rank], rank


def _centroid_jacobian(model, inputs, targets):
    rows = []
    for label in range(CLASS_COUNT):
        x = inputs[targets == label]
        z0 = x @ model["w0"] + model["b0"]
        h0 = np.maximum(z0, 0.0)
        z1 = h0 @ model["w1"] + model["b1"]
        gate0, gate1 = z0 > 0, z1 > 0
        for output in range(model["w1"].shape[1]):
            cross = x.T @ (gate1[:, output, None] * gate0) / len(x)
            w0_row = (cross * model["w1"][:, output][None, :]).ravel()
            b0_row = (gate1[:, output, None] * gate0).mean(axis=0) * model["w1"][:, output]
            w1_block = np.zeros_like(model["w1"])
            w1_block[:, output] = (h0 * gate1[:, output, None]).mean(axis=0)
            b1_row = np.zeros_like(model["b1"])
            b1_row[output] = gate1[:, output].mean()
            rows.append(np.concatenate([w0_row, b0_row, w1_block.ravel(), b1_row]))
    return np.asarray(rows).reshape(CLASS_COUNT, 32, -1)


def _pair_constraints(model, inputs, targets, first, second):
    means, hidden_direction, decision_direction, basis, rank = _pair_subspace(model, inputs, targets, first, second)
    centroid_jacobian = _centroid_jacobian(model, inputs, targets)
    directions, gains, labels, kinds = [], [], [], []
    for label, sign in ((first, 1.0), (second, -1.0)):
        directions.append(hidden_direction)
        gains.append(sign * PAIR_GAIN)
        labels.append(label)
        kinds.append("hidden_pair_subspace")
        directions.append(decision_direction if sign > 0 else -decision_direction)
        gains.append(PAIR_GAIN)
        labels.append(label)
        kinds.append("decision_direction")
    centroids = np.asarray(means)
    logits = centroids @ model["w2"] + model["b2"]
    for label in range(CLASS_COUNT):
        if label in (first, second):
            continue
        order = np.argsort(logits[label])[::-1]
        competitor = int(order[0] if order[0] != label else order[1])
        directions.append(model["w2"][:, label] - model["w2"][:, competitor])
        gains.append(0.0)
        labels.append(label)
        kinds.append("other_class_margin_preserve")
    jacobian = np.asarray([np.einsum("j,jp->p", direction, centroid_jacobian[label]) for direction, label in zip(directions, labels)])
    return jacobian, np.asarray(gains), {"pair": [first, second], "pair_confusion": None, "subspace_rank": rank, "hidden_direction": hidden_direction.tolist(), "decision_direction": decision_direction.tolist(), "constraint_kinds": kinds, "constraint_labels": labels, "basis": basis.tolist()}


def _pair_metrics(model, inputs, targets, first, second, basis):
    h2 = _forward(model, inputs, 2)[0][-1]
    means = np.asarray([h2[targets == label].mean(axis=0) for label in range(CLASS_COUNT)])
    projections = h2 @ basis
    pair_mean_gap = means[first] @ basis - means[second] @ basis
    pooled = np.sqrt(0.5 * (projections[targets == first].var(axis=0) + projections[targets == second].var(axis=0)))
    separation = float(np.linalg.norm(pair_mean_gap) / max(np.linalg.norm(pooled), 1e-12))
    logits = h2 @ model["w2"] + model["b2"]
    pair_margins = np.concatenate([logits[targets == first, first] - logits[targets == first, second], logits[targets == second, second] - logits[targets == second, first]])
    predictions = logits.argmax(axis=1)
    pair_mask = np.isin(targets, (first, second))
    pair_recall = float(np.mean(predictions[pair_mask] == targets[pair_mask]))
    return {"subspace_separation": separation, "mean_pair_margin": float(pair_margins.mean()), "pair_recall": pair_recall, "pair_mean_gap_norm": float(np.linalg.norm(pair_mean_gap))}


def _evaluate(model, inputs, targets, first, second, basis, baseline_predictions=None):
    margins = _margin_metrics(model, inputs, targets)
    predictions = _forward(model, inputs, 2)[2].argmax(axis=1)
    other = ~np.isin(targets, (first, second))
    result = {"margin": margins, "pair": _pair_metrics(model, inputs, targets, first, second, basis), "geometry": _geometry(model, inputs, targets), "other_class_accuracy": float(np.mean(predictions[other] == targets[other]))}
    if baseline_predictions is not None:
        result["other_class_preservation"] = float(np.mean(predictions[other] == baseline_predictions[other]))
    return result


def run_seed(data, seed):
    model = _train_model(_init_model(561, (64, 32), CLASS_COUNT, seed), data["train_x"], data["train_y"], seed)
    validation_baseline = _classification(model, data["val_x"], data["val_y"])
    pair, confusion, pair_confusion = _confusion_pair(model, data["val_x"], data["val_y"])
    first, second = pair
    _, _, _, basis, rank = _pair_subspace(model, data["val_x"], data["val_y"], first, second)
    selective = _selective_constraints(model, data["val_x"], data["val_y"], _partition(model, data["val_x"], data["val_y"]))
    selective_delta = _inverse_delta(selective[-1], selective[2])
    pair_jacobian, pair_gains, pair_info = _pair_constraints(model, data["val_x"], data["val_y"], first, second)
    pair_delta = _inverse_delta(pair_jacobian, pair_gains)
    gradient = _gradients(model, data["train_x"], data["train_y"], 2)[3]
    gradient_vector = np.concatenate([gradient["w0"].ravel(), gradient["b0"], gradient["w1"].ravel(), gradient["b1"]])
    rng = np.random.default_rng(seed + 9300)
    pair_norm = np.linalg.norm(pair_delta)
    candidates = {"sample_selective_inverse": _apply_hidden_delta(model, selective_delta), "class_pair_inverse": _apply_hidden_delta(model, pair_delta), "random_same_norm": _apply_hidden_delta(model, _same_norm_delta(pair_delta, rng)), "gradient_same_norm": _apply_hidden_delta(model, -gradient_vector * (pair_norm / max(np.linalg.norm(gradient_vector), 1e-12)))}
    baseline_val_predictions = _forward(model, data["val_x"], 2)[2].argmax(axis=1)
    candidate_results = {}
    for name, candidate in candidates.items():
        candidate_results[name] = {"validation": _evaluate(candidate, data["val_x"], data["val_y"], first, second, basis, baseline_val_predictions), "safe_validation": _safe_preservation(model, candidate, data["val_x"], data["val_y"], _partition(model, data["val_x"], data["val_y"])["safe"]), "delta_norm": float(np.linalg.norm(_hidden_vector(candidate) - _hidden_vector(model))), "gate_change_fraction": float(np.mean(_gate_state(candidate, data["val_x"]) != _gate_state(model, data["val_x"]))), "model": candidate}
    current = _copy(model)
    iterative = []
    partition = _partition(model, data["val_x"], data["val_y"])
    for iteration in range(ITERATIONS):
        jacobian, gains, _ = _pair_constraints(current, data["val_x"], data["val_y"], first, second)
        step = _inverse_delta(jacobian, gains)
        current = _apply_hidden_delta(current, step, STEP_FRACTION)
        iterative.append({"iteration": iteration + 1, "validation": _evaluate(current, data["val_x"], data["val_y"], first, second, basis, baseline_val_predictions), "safe_validation": _safe_preservation(model, current, data["val_x"], data["val_y"], partition["safe"]), "step_norm": float(np.linalg.norm(step) * STEP_FRACTION), "model": _copy(current)})
    baseline_test_predictions = _forward(model, data["test_x"], 2)[2].argmax(axis=1)
    baseline_test = _evaluate(model, data["test_x"], data["test_y"], first, second, basis, baseline_test_predictions)
    for result in candidate_results.values():
        candidate = result.pop("model")
        result["test"] = _evaluate(candidate, data["test_x"], data["test_y"], first, second, basis, baseline_test_predictions)
    for result in iterative:
        candidate = result.pop("model")
        result["test"] = _evaluate(candidate, data["test_x"], data["test_y"], first, second, basis, baseline_test_predictions)
    pair_info.update({"pair": list(pair), "pair_confusion": pair_confusion, "validation_confusion": confusion})
    return {"seed": seed, "pair_info": pair_info, "validation_baseline": validation_baseline, "baseline_test": baseline_test, "one_shot": candidate_results, "iterative": iterative, "pair_jacobian_shape": list(pair_jacobian.shape), "subspace_rank": rank}


def run_class_pair_inverse(data_dir, seeds=(7, 11, 19, 23, 31)):
    data = _strict_split(data_dir)
    runs = [run_seed(data, seed) for seed in seeds]
    names = ("sample_selective_inverse", "class_pair_inverse", "random_same_norm", "gradient_same_norm")
    summary = {"pair_frequency": {str(pair): int(sum(tuple(run["pair_info"]["pair"]) == pair for run in runs)) for pair in sorted({tuple(run["pair_info"]["pair"]) for run in runs})}, "baseline": {"accuracy": float(np.mean([r["baseline_test"]["margin"]["accuracy"] for r in runs]))}, "one_shot": {name: {"accuracy": float(np.mean([r["one_shot"][name]["test"]["margin"]["accuracy"] for r in runs])), "pair_recall": float(np.mean([r["one_shot"][name]["test"]["pair"]["pair_recall"] for r in runs])), "pair_margin": float(np.mean([r["one_shot"][name]["test"]["pair"]["mean_pair_margin"] for r in runs])), "subspace_separation": float(np.mean([r["one_shot"][name]["test"]["pair"]["subspace_separation"] for r in runs])), "other_class_accuracy": float(np.mean([r["one_shot"][name]["test"]["other_class_accuracy"] for r in runs])), "other_class_preservation": float(np.mean([r["one_shot"][name]["test"]["other_class_preservation"] for r in runs])), "gate_change_fraction": float(np.mean([r["one_shot"][name]["gate_change_fraction"] for r in runs]))} for name in names}, "iterative": {"accuracy": [float(np.mean([r["iterative"][i]["test"]["margin"]["accuracy"] for r in runs])) for i in range(ITERATIONS)], "pair_recall": [float(np.mean([r["iterative"][i]["test"]["pair"]["pair_recall"] for r in runs])) for i in range(ITERATIONS)], "subspace_separation": [float(np.mean([r["iterative"][i]["test"]["pair"]["subspace_separation"] for r in runs])) for i in range(ITERATIONS)]}}
    return {"settings": {"architecture": "561→64→32→6", "seeds": list(seeds), "split": "strict train/validation/test", "pair_gain": PAIR_GAIN}, "summary": summary, "runs": runs}


def write_report(result, path):
    s = result["summary"]
    baseline_pair_recall = np.mean([run["baseline_test"]["pair"]["pair_recall"] for run in result["runs"]])
    baseline_pair_margin = np.mean([run["baseline_test"]["pair"]["mean_pair_margin"] for run in result["runs"]])
    baseline_separation = np.mean([run["baseline_test"]["pair"]["subspace_separation"] for run in result["runs"]])
    lines = ["# Validation class-pair decision subspace inverse", "", "validation confusion에서 반복 혼동 pair를 찾고, 해당 pair의 hidden mean direction과 output decision direction을 함께 제약했다. test는 모든 update 후 최종 평가했다.", "", f"pair frequency: {s['pair_frequency']}", f"baseline test: accuracy={s['baseline']['accuracy']:.3f}, pair recall={baseline_pair_recall:.3f}, pair margin={baseline_pair_margin:.3f}, subspace separation={baseline_separation:.3f}", "", "## Test 결과"]
    for name, value in s["one_shot"].items():
        lines.append(f"- {name}: accuracy={value['accuracy']:.3f}, pair recall={value['pair_recall']:.3f}, pair margin={value['pair_margin']:.3f}, subspace separation={value['subspace_separation']:.3f}, other accuracy={value['other_class_accuracy']:.3f}, other preservation={value['other_class_preservation']:.3f}, gate change={value['gate_change_fraction']:.1%}")
    lines += ["", "## 반복 pair inverse", f"- accuracy: {', '.join(f'{x:.3f}' for x in s['iterative']['accuracy'])}", f"- pair recall: {', '.join(f'{x:.3f}' for x in s['iterative']['pair_recall'])}", f"- subspace separation: {', '.join(f'{x:.3f}' for x in s['iterative']['subspace_separation'])}", "", "## 해석", "", "class-level inverse는 pair margin과 hidden subspace separation은 증가시켰지만, discrete pair recall은 baseline보다 높아지지 않았다. 따라서 class-level geometry의 일부는 test에 일반화됐지만, decision threshold를 넘는 새로운 정답 예측으로 이어지지는 않았다."]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    root = Path(__file__).parent
    result = run_class_pair_inverse(root / "UCI HAR Dataset")
    (root / "class_pair_inverse_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_report(result, root / "class_pair_inverse_analysis.md")
    print(json.dumps({"runs": len(result["runs"]), "result": str(root / "class_pair_inverse_results.json")}, indent=2))
