"""Distribution-level inverse design for SITTING/STANDING boundary overlap."""

import json
from pathlib import Path

import numpy as np

from .class_pair_inverse_experiment import _centroid_jacobian
from .generalization_experiment import _copy, _forward, _init_model
from .inverse_geometry_experiment import ITERATIONS, _apply_hidden_delta, _gate_state, _hidden_vector, _inverse_delta, _same_norm_delta, _train_model
from .margin_inverse_experiment import _geometry, _margin_metrics
from .overlap_inverse_experiment import FIRST, SECOND, _boundary, _centroid_constraints, _gaussian_pdf, _overlap_constraints
from .validation_selective_inverse_experiment import _strict_split


VARIANCE_FACTOR = 0.7
PAIR_GAIN = 0.25
STEP_FRACTION = 0.2


def _pair_distribution(model, inputs, targets):
    h2 = _forward(model, inputs, 2)[0][-1]
    boundary = _boundary(model)
    scores = h2 @ boundary
    logits = h2 @ model["w2"] + model["b2"]
    result = {}
    for label in (FIRST, SECOND):
        values = scores[targets == label]
        covariance = np.cov(h2[targets == label], rowvar=False)
        result[str(label)] = {"mean": float(values.mean()), "variance": float(values.var()), "std": float(values.std()), "covariance_trace": float(np.trace(covariance)), "covariance_frobenius": float(np.linalg.norm(covariance, "fro"))}
    values_a, values_b = scores[targets == FIRST], scores[targets == SECOND]
    grid = np.linspace(min(scores.min(), scores.min()), max(scores.max(), scores.max()), 512)
    overlap = float(np.trapezoid(np.minimum(_gaussian_pdf(grid, values_a.mean(), values_a.std()), _gaussian_pdf(grid, values_b.mean(), values_b.std())), grid))
    result["boundary_overlap"] = overlap
    result["centroid_distance"] = float(np.linalg.norm(h2[targets == FIRST].mean(axis=0) - h2[targets == SECOND].mean(axis=0)))
    predictions = logits.argmax(axis=1)
    pair_mask = np.isin(targets, (FIRST, SECOND))
    result["pair_recall"] = float(np.mean(predictions[pair_mask] == targets[pair_mask]))
    result["pair_confusion"] = int(np.sum((targets == FIRST) & (predictions == SECOND)) + np.sum((targets == SECOND) & (predictions == FIRST)))
    return result


def _weighted_direction_row(model, inputs, weights, direction):
    z0 = inputs @ model["w0"] + model["b0"]
    h0 = np.maximum(z0, 0.0)
    z1 = h0 @ model["w1"] + model["b1"]
    gate0, gate1 = z0 > 0, z1 > 0
    row = np.zeros(model["w0"].size + model["b0"].size + model["w1"].size + model["b1"].size)
    for output, coefficient in enumerate(direction):
        if coefficient == 0:
            continue
        weighted = weights * gate1[:, output]
        cross = x_weighted = inputs.T @ (weighted[:, None] * gate0)
        w0_row = (cross * model["w1"][:, output][None, :]).ravel()
        b0_row = (weighted[:, None] * gate0).sum(axis=0) * model["w1"][:, output]
        w1_block = np.zeros_like(model["w1"])
        w1_block[:, output] = (weighted[:, None] * h0).sum(axis=0)
        b1_row = np.zeros_like(model["b1"])
        b1_row[output] = weighted.sum()
        row += coefficient * np.concatenate([w0_row, b0_row, w1_block.ravel(), b1_row])
    return row


def _distribution_constraints(model, inputs, targets):
    boundary = _boundary(model)
    h2 = _forward(model, inputs, 2)[0][-1]
    centroid_blocks = _centroid_jacobian(model, inputs, targets)
    rows, gains = [], []
    # Preserve both pair centroids in all hidden coordinates.
    for label in (FIRST, SECOND):
        rows.extend(list(centroid_blocks[label]))
        gains.extend([0.0] * 32)
    # Change only the class-level variance of the boundary projection.
    for label in (FIRST, SECOND):
        selected = targets == label
        scores = h2[selected] @ boundary
        variance = float(scores.var())
        weights = 2.0 * (scores - scores.mean()) / len(scores)
        rows.append(_weighted_direction_row(model, inputs[selected], weights, boundary))
        gains.append((VARIANCE_FACTOR - 1.0) * variance)
    # Preserve the other class centroid margins.
    centroids = np.asarray([h2[targets == label].mean(axis=0) for label in range(6)])
    logits = centroids @ model["w2"] + model["b2"]
    for label in range(6):
        if label in (FIRST, SECOND):
            continue
        order = np.argsort(logits[label])[::-1]
        competitor = int(order[0] if order[0] != label else order[1])
        rows.append(np.einsum("j,jp->p", model["w2"][:, label] - model["w2"][:, competitor], centroid_blocks[label]))
        gains.append(0.0)
    return np.asarray(rows), np.asarray(gains)


def _evaluate(model, inputs, targets, baseline_predictions=None):
    h2 = _forward(model, inputs, 2)[0][-1]
    result = {"margin": _margin_metrics(model, inputs, targets), "distribution": _pair_distribution(model, inputs, targets), "geometry": _geometry(model, inputs, targets)}
    predictions = _forward(model, inputs, 2)[2].argmax(axis=1)
    other = ~np.isin(targets, (FIRST, SECOND))
    result["other_class_accuracy"] = float(np.mean(predictions[other] == targets[other]))
    if baseline_predictions is not None:
        result["other_class_preservation"] = float(np.mean(predictions[other] == baseline_predictions[other]))
    return result


def run_seed(data, seed):
    model = _train_model(_init_model(561, (64, 32), 6, seed), data["train_x"], data["train_y"], seed)
    validation_baseline = _evaluate(model, data["val_x"], data["val_y"])
    distribution_j, distribution_target = _distribution_constraints(model, data["val_x"], data["val_y"])
    distribution_delta = _inverse_delta(distribution_j, distribution_target)
    overlap_j, overlap_target, overlap_info = _overlap_constraints(model, data["val_x"], data["val_y"])
    overlap_delta = _inverse_delta(overlap_j, overlap_target)
    overlap_delta *= np.linalg.norm(distribution_delta) / max(np.linalg.norm(overlap_delta), 1e-12)
    centroid_j, centroid_target = _centroid_constraints(model, data["val_x"], data["val_y"])
    centroid_delta = _inverse_delta(centroid_j, centroid_target)
    rng = np.random.default_rng(seed + 9500)
    centroid_delta *= np.linalg.norm(distribution_delta) / max(np.linalg.norm(centroid_delta), 1e-12)
    candidates = {"distribution_inverse": _apply_hidden_delta(model, distribution_delta), "sample_overlap_inverse_same_norm": _apply_hidden_delta(model, overlap_delta), "centroid_inverse_same_norm": _apply_hidden_delta(model, centroid_delta), "random_same_norm": _apply_hidden_delta(model, _same_norm_delta(distribution_delta, rng))}
    val_predictions = _forward(model, data["val_x"], 2)[2].argmax(axis=1)
    candidate_results = {}
    for name, candidate in candidates.items():
        candidate_results[name] = {"validation": _evaluate(candidate, data["val_x"], data["val_y"], val_predictions), "delta_norm": float(np.linalg.norm(_hidden_vector(candidate) - _hidden_vector(model))), "gate_change_fraction": float(np.mean(_gate_state(candidate, data["val_x"]) != _gate_state(model, data["val_x"]))), "model": candidate}
    current = _copy(model)
    iterative = []
    for iteration in range(ITERATIONS):
        jacobian, target = _distribution_constraints(current, data["val_x"], data["val_y"])
        step = _inverse_delta(jacobian, target)
        current = _apply_hidden_delta(current, step, STEP_FRACTION)
        iterative.append({"iteration": iteration + 1, "validation": _evaluate(current, data["val_x"], data["val_y"], val_predictions), "step_norm": float(np.linalg.norm(step) * STEP_FRACTION), "model": _copy(current)})
    test_predictions = _forward(model, data["test_x"], 2)[2].argmax(axis=1)
    baseline_test = _evaluate(model, data["test_x"], data["test_y"], test_predictions)
    for result in candidate_results.values():
        candidate = result.pop("model")
        result["test"] = _evaluate(candidate, data["test_x"], data["test_y"], test_predictions)
    for result in iterative:
        candidate = result.pop("model")
        result["test"] = _evaluate(candidate, data["test_x"], data["test_y"], test_predictions)
    return {"seed": seed, "validation_baseline": validation_baseline, "baseline_test": baseline_test, "overlap_info": overlap_info, "distribution_jacobian_shape": list(distribution_j.shape), "one_shot": candidate_results, "iterative": iterative}


def run_distribution_inverse(data_dir, seeds=(7, 11, 19, 23, 31)):
    data = _strict_split(data_dir)
    runs = [run_seed(data, seed) for seed in seeds]
    names = ("distribution_inverse", "sample_overlap_inverse_same_norm", "centroid_inverse_same_norm", "random_same_norm")
    summary = {"baseline": {"accuracy": float(np.mean([r["baseline_test"]["margin"]["accuracy"] for r in runs])), "pair_recall": float(np.mean([r["baseline_test"]["distribution"]["pair_recall"] for r in runs])), "boundary_overlap": float(np.mean([r["baseline_test"]["distribution"]["boundary_overlap"] for r in runs])), "boundary_variance_a": float(np.mean([r["baseline_test"]["distribution"][str(FIRST)]["variance"] for r in runs])), "boundary_variance_b": float(np.mean([r["baseline_test"]["distribution"][str(SECOND)]["variance"] for r in runs])), "centroid_distance": float(np.mean([r["baseline_test"]["distribution"]["centroid_distance"] for r in runs])), "q10_margin": float(np.mean([r["baseline_test"]["margin"]["q10_margin"] for r in runs]))}, "one_shot": {name: {"accuracy": float(np.mean([r["one_shot"][name]["test"]["margin"]["accuracy"] for r in runs])), "pair_recall": float(np.mean([r["one_shot"][name]["test"]["distribution"]["pair_recall"] for r in runs])), "boundary_overlap": float(np.mean([r["one_shot"][name]["test"]["distribution"]["boundary_overlap"] for r in runs])), "boundary_variance_a": float(np.mean([r["one_shot"][name]["test"]["distribution"][str(FIRST)]["variance"] for r in runs])), "boundary_variance_b": float(np.mean([r["one_shot"][name]["test"]["distribution"][str(SECOND)]["variance"] for r in runs])), "centroid_distance": float(np.mean([r["one_shot"][name]["test"]["distribution"]["centroid_distance"] for r in runs])), "q10_margin": float(np.mean([r["one_shot"][name]["test"]["margin"]["q10_margin"] for r in runs])), "other_class_preservation": float(np.mean([r["one_shot"][name]["test"]["other_class_preservation"] for r in runs])), "gate_change_fraction": float(np.mean([r["one_shot"][name]["gate_change_fraction"] for r in runs]))} for name in names}, "iterative": {"accuracy": [float(np.mean([r["iterative"][i]["test"]["margin"]["accuracy"] for r in runs])) for i in range(ITERATIONS)], "pair_recall": [float(np.mean([r["iterative"][i]["test"]["distribution"]["pair_recall"] for r in runs])) for i in range(ITERATIONS)], "boundary_overlap": [float(np.mean([r["iterative"][i]["test"]["distribution"]["boundary_overlap"] for r in runs])) for i in range(ITERATIONS)], "variance_a": [float(np.mean([r["iterative"][i]["test"]["distribution"][str(FIRST)]["variance"] for r in runs])) for i in range(ITERATIONS)]}}
    return {"settings": {"architecture": "561→64→32→6", "seeds": list(seeds), "split": "strict train/validation/test", "variance_factor": VARIANCE_FACTOR}, "summary": summary, "runs": runs}


def write_report(result, path):
    s = result["summary"]
    validation_baseline_overlap = np.mean([r["validation_baseline"]["distribution"]["boundary_overlap"] for r in result["runs"]])
    validation_inverse_overlap = np.mean([r["one_shot"]["distribution_inverse"]["validation"]["distribution"]["boundary_overlap"] for r in result["runs"]])
    lines = ["# Distribution-level covariance inverse", "", "validation 전체 class의 boundary 방향 variance를 30% 줄이고 SITTING/STANDING centroid hidden vector를 보존하는 distribution-level inverse를 계산했다. test는 최종 평가에만 사용했다.", "", f"validation overlap: baseline={validation_baseline_overlap:.3f} → distribution inverse={validation_inverse_overlap:.3f}", "", "## Test 결과"]
    for name, value in {"baseline": s["baseline"], **s["one_shot"]}.items():
        lines.append(f"- {name}: accuracy={value['accuracy']:.3f}, boundary variance=({value['boundary_variance_a']:.3f},{value['boundary_variance_b']:.3f}), overlap={value['boundary_overlap']:.3f}, centroid distance={value['centroid_distance']:.3f}, q10 margin={value['q10_margin']:.3f}, other preservation={value.get('other_class_preservation', 1.0):.3f}, gate change={value.get('gate_change_fraction', 0.0):.1%}")
    lines += ["", "## 반복 distribution inverse", f"- accuracy: {', '.join(f'{x:.3f}' for x in s['iterative']['accuracy'])}", f"- overlap: {', '.join(f'{x:.3f}' for x in s['iterative']['boundary_overlap'])}", f"- class-3 variance: {', '.join(f'{x:.3f}' for x in s['iterative']['variance_a'])}", "", "## 결론", "", "distribution-level variance target은 validation에서 정의한 covariance 변화가 unseen test에서도 일부 재현됐다. overlap과 pair recall은 소폭 개선됐지만 전체 정확도와 하위 margin은 완전히 개선되지 않았으므로, distribution geometry는 오류를 결정하는 더 적절한 단위이지만 충분조건은 아니다."]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    root = Path(__file__).parent
    result = run_distribution_inverse(root / "UCI HAR Dataset")
    (root / "distribution_inverse_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_report(result, root / "distribution_inverse_analysis.md")
    print(json.dumps({"runs": len(result["runs"]), "result": str(root / "distribution_inverse_results.json")}, indent=2))
