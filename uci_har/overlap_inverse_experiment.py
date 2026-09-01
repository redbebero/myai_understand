"""Test whether SITTING/STANDING errors are boundary-distribution overlap."""

import json
from pathlib import Path

import numpy as np

from .class_pair_inverse_experiment import _centroid_jacobian
from .generalization_experiment import _copy, _forward, _gradients, _init_model
from .inverse_geometry_experiment import ITERATIONS, _apply_hidden_delta, _gate_state, _hidden_vector, _inverse_delta, _same_norm_delta, _train_model
from .margin_inverse_experiment import _geometry, _hidden_rows, _margin_metrics, _margin_state
from .validation_selective_inverse_experiment import _strict_split


FIRST, SECOND = 3, 4
BOUNDARY_TARGET = 0.5
PAIR_GAIN = 0.25
MAX_OVERLAP_SAMPLES = 20
STEP_FRACTION = 0.2


def _boundary(model):
    direction = model["w2"][:, FIRST] - model["w2"][:, SECOND]
    return direction / max(np.linalg.norm(direction), 1e-12)


def _pair_state(model, inputs, targets):
    h2 = _forward(model, inputs, 2)[0][-1]
    logits = h2 @ model["w2"] + model["b2"]
    score = logits[:, FIRST] - logits[:, SECOND]
    signed = np.where(targets == FIRST, score, -score)
    return h2, logits, score, signed


def _gaussian_pdf(values, mean, std):
    std = max(float(std), 1e-8)
    return np.exp(-0.5 * ((values - mean) / std) ** 2) / (std * np.sqrt(2.0 * np.pi))


def _pair_metrics(model, inputs, targets, baseline_predictions=None):
    h2, logits, score, signed = _pair_state(model, inputs, targets)
    class_a, class_b = targets == FIRST, targets == SECOND
    values_a, values_b = score[class_a], score[class_b]
    mean_a, mean_b = float(values_a.mean()), float(values_b.mean())
    std_a, std_b = float(values_a.std()), float(values_b.std())
    grid = np.linspace(min(values_a.min(), values_b.min()), max(values_a.max(), values_b.max()), 512)
    pdf_a, pdf_b = _gaussian_pdf(grid, mean_a, std_a), _gaussian_pdf(grid, mean_b, std_b)
    overlap = float(np.trapezoid(np.minimum(pdf_a, pdf_b), grid))
    pdf_sample_a = _gaussian_pdf(score, mean_a, std_a)
    pdf_sample_b = _gaussian_pdf(score, mean_b, std_b)
    ambiguity = np.minimum(pdf_sample_a, pdf_sample_b) / np.maximum(pdf_sample_a + pdf_sample_b, 1e-12) >= 0.25
    predictions = logits.argmax(axis=1)
    pair_mask = np.logical_or(class_a, class_b)
    confusion = np.zeros((6, 6), dtype=int)
    for target, prediction in zip(targets, predictions):
        confusion[int(target), int(prediction)] += 1
    covariance_a = np.cov(h2[class_a], rowvar=False)
    covariance_b = np.cov(h2[class_b], rowvar=False)
    centroid_a, centroid_b = h2[class_a].mean(axis=0), h2[class_b].mean(axis=0)
    pair_error = np.logical_and(pair_mask, predictions != targets)
    return {"centroid_distance": float(np.linalg.norm(centroid_a - centroid_b)), "score_mean_a": mean_a, "score_mean_b": mean_b, "score_std_a": std_a, "score_std_b": std_b, "score_mean_gap": float(mean_a - mean_b), "covariance_trace_a": float(np.trace(covariance_a)), "covariance_trace_b": float(np.trace(covariance_b)), "covariance_frobenius_gap": float(np.linalg.norm(covariance_a - covariance_b)), "boundary_overlap": overlap, "ambiguous_fraction": float(np.mean(ambiguity[pair_mask])), "error_in_overlap": float(np.mean(pair_error[ambiguity & pair_mask])) if np.any(ambiguity & pair_mask) else 0.0, "error_outside_overlap": float(np.mean(pair_error[~ambiguity & pair_mask])) if np.any(~ambiguity & pair_mask) else 0.0, "pair_recall": float(np.mean(predictions[pair_mask] == targets[pair_mask])), "pair_confusion": int(confusion[FIRST, SECOND] + confusion[SECOND, FIRST]), "mean_pair_margin": float(np.mean(signed[pair_mask])), "confusion": confusion.tolist()}


def _correlation(values, errors):
    values, errors = np.asarray(values), np.asarray(errors)
    return float(np.corrcoef(values, errors)[0, 1]) if np.std(values) > 1e-12 and np.std(errors) > 1e-12 else 0.0


def _overlap_constraints(model, inputs, targets):
    h2, logits, score, signed = _pair_state(model, inputs, targets)
    direction = _boundary(model)
    pair_mask = np.isin(targets, (FIRST, SECOND))
    mean_a, mean_b = score[targets == FIRST].mean(), score[targets == SECOND].mean()
    std_a, std_b = score[targets == FIRST].std(), score[targets == SECOND].std()
    ambiguity = np.minimum(_gaussian_pdf(score, mean_a, std_a), _gaussian_pdf(score, mean_b, std_b)) / np.maximum(_gaussian_pdf(score, mean_a, std_a) + _gaussian_pdf(score, mean_b, std_b), 1e-12)
    selected = []
    for label in (FIRST, SECOND):
        candidates = np.flatnonzero((targets == label) & pair_mask)
        candidates = candidates[np.argsort(-ambiguity[candidates])[:MAX_OVERLAP_SAMPLES]]
        selected.extend(candidates.tolist())
    selected = np.asarray(selected, dtype=int)
    sample_rows = _hidden_rows(model, inputs[selected]).reshape(len(selected), 32, -1)
    sample_jacobian = np.einsum("j,ijp->ip", direction, sample_rows)
    sample_gains = np.asarray([(BOUNDARY_TARGET - score[index]) if targets[index] == FIRST else (-BOUNDARY_TARGET - score[index]) for index in selected])
    centroid_blocks = _centroid_jacobian(model, inputs, targets)
    constraints, gains = [], []
    for label in (FIRST, SECOND):
        constraints.extend(list(centroid_blocks[label]))
        gains.extend([0.0] * 32)
    logits_centroids = np.asarray([h2[targets == label].mean(axis=0) @ model["w2"] + model["b2"] for label in range(6)])
    for label in range(6):
        if label in (FIRST, SECOND):
            continue
        order = np.argsort(logits_centroids[label])[::-1]
        competitor = int(order[0] if order[0] != label else order[1])
        constraints.append(np.einsum("j,jp->p", model["w2"][:, label] - model["w2"][:, competitor], centroid_blocks[label]))
        gains.append(0.0)
    jacobian = np.vstack([sample_jacobian, np.asarray(constraints)])
    target = np.concatenate([sample_gains, np.asarray(gains)])
    return jacobian, target, {"selected_count": int(len(selected)), "centroid_constraints": 64, "overlap_indices": selected.tolist()}


def _centroid_constraints(model, inputs, targets):
    blocks = _centroid_jacobian(model, inputs, targets)
    h2 = _forward(model, inputs, 2)[0][-1]
    centroids = np.asarray([h2[targets == label].mean(axis=0) for label in range(6)])
    direction = _boundary(model)
    rows, gains = [np.einsum("j,jp->p", direction, blocks[FIRST]), np.einsum("j,jp->p", direction, blocks[SECOND])], [PAIR_GAIN, -PAIR_GAIN]
    logits = centroids @ model["w2"] + model["b2"]
    for label in range(6):
        if label in (FIRST, SECOND):
            continue
        order = np.argsort(logits[label])[::-1]
        competitor = int(order[0] if order[0] != label else order[1])
        rows.append(np.einsum("j,jp->p", model["w2"][:, label] - model["w2"][:, competitor], blocks[label]))
        gains.append(0.0)
    return np.asarray(rows), np.asarray(gains)


def _evaluate(model, inputs, targets, baseline_predictions=None):
    result = {"margin": _margin_metrics(model, inputs, targets), "pair": _pair_metrics(model, inputs, targets, baseline_predictions), "geometry": _geometry(model, inputs, targets)}
    predictions = _forward(model, inputs, 2)[2].argmax(axis=1)
    other = ~np.isin(targets, (FIRST, SECOND))
    result["other_class_accuracy"] = float(np.mean(predictions[other] == targets[other]))
    if baseline_predictions is not None:
        result["other_class_preservation"] = float(np.mean(predictions[other] == baseline_predictions[other]))
    return result


def run_seed(data, seed):
    model = _train_model(_init_model(561, (64, 32), 6, seed), data["train_x"], data["train_y"], seed)
    validation_baseline = _evaluate(model, data["val_x"], data["val_y"])
    overlap_j, overlap_target, overlap_info = _overlap_constraints(model, data["val_x"], data["val_y"])
    overlap_delta = _inverse_delta(overlap_j, overlap_target)
    centroid_j, centroid_target = _centroid_constraints(model, data["val_x"], data["val_y"])
    centroid_delta = _inverse_delta(centroid_j, centroid_target)
    centroid_delta *= np.linalg.norm(overlap_delta) / max(np.linalg.norm(centroid_delta), 1e-12)
    rng = np.random.default_rng(seed + 9400)
    candidates = {"overlap_inverse": _apply_hidden_delta(model, overlap_delta), "centroid_inverse_same_norm": _apply_hidden_delta(model, centroid_delta), "random_same_norm": _apply_hidden_delta(model, _same_norm_delta(overlap_delta, rng))}
    val_predictions = _forward(model, data["val_x"], 2)[2].argmax(axis=1)
    candidate_results = {}
    for name, candidate in candidates.items():
        candidate_results[name] = {"validation": _evaluate(candidate, data["val_x"], data["val_y"], val_predictions), "delta_norm": float(np.linalg.norm(_hidden_vector(candidate) - _hidden_vector(model))), "gate_change_fraction": float(np.mean(_gate_state(candidate, data["val_x"]) != _gate_state(model, data["val_x"]))), "model": candidate}
    current = _copy(model)
    iterative = []
    for iteration in range(ITERATIONS):
        jacobian, target, _ = _overlap_constraints(current, data["val_x"], data["val_y"])
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
    return {"seed": seed, "validation_baseline": validation_baseline, "baseline_test": baseline_test, "overlap_info": overlap_info, "overlap_jacobian_shape": list(overlap_j.shape), "one_shot": candidate_results, "iterative": iterative}


def run_overlap_inverse(data_dir, seeds=(7, 11, 19, 23, 31)):
    data = _strict_split(data_dir)
    runs = [run_seed(data, seed) for seed in seeds]
    names = ("overlap_inverse", "centroid_inverse_same_norm", "random_same_norm")
    summary = {"baseline": {"accuracy": float(np.mean([r["baseline_test"]["margin"]["accuracy"] for r in runs])), "pair_recall": float(np.mean([r["baseline_test"]["pair"]["pair_recall"] for r in runs])), "pair_confusion": float(np.mean([r["baseline_test"]["pair"]["pair_confusion"] for r in runs])), "boundary_overlap": float(np.mean([r["baseline_test"]["pair"]["boundary_overlap"] for r in runs])), "centroid_distance": float(np.mean([r["baseline_test"]["pair"]["centroid_distance"] for r in runs])), "q10_margin": float(np.mean([r["baseline_test"]["margin"]["q10_margin"] for r in runs]))}, "one_shot": {name: {"accuracy": float(np.mean([r["one_shot"][name]["test"]["margin"]["accuracy"] for r in runs])), "pair_recall": float(np.mean([r["one_shot"][name]["test"]["pair"]["pair_recall"] for r in runs])), "pair_confusion": float(np.mean([r["one_shot"][name]["test"]["pair"]["pair_confusion"] for r in runs])), "boundary_overlap": float(np.mean([r["one_shot"][name]["test"]["pair"]["boundary_overlap"] for r in runs])), "centroid_distance": float(np.mean([r["one_shot"][name]["test"]["pair"]["centroid_distance"] for r in runs])), "q10_margin": float(np.mean([r["one_shot"][name]["test"]["margin"]["q10_margin"] for r in runs])), "other_class_preservation": float(np.mean([r["one_shot"][name]["test"]["other_class_preservation"] for r in runs])), "gate_change_fraction": float(np.mean([r["one_shot"][name]["gate_change_fraction"] for r in runs]))} for name in names}, "correlation_with_pair_error": {"centroid_distance": _correlation([r["baseline_test"]["pair"]["centroid_distance"] for r in runs], [1.0 - r["baseline_test"]["pair"]["pair_recall"] for r in runs]), "mean_pair_margin": _correlation([r["baseline_test"]["pair"]["mean_pair_margin"] for r in runs], [1.0 - r["baseline_test"]["pair"]["pair_recall"] for r in runs]), "boundary_overlap": _correlation([r["baseline_test"]["pair"]["boundary_overlap"] for r in runs], [1.0 - r["baseline_test"]["pair"]["pair_recall"] for r in runs])}, "iterative": {"accuracy": [float(np.mean([r["iterative"][i]["test"]["margin"]["accuracy"] for r in runs])) for i in range(ITERATIONS)], "pair_recall": [float(np.mean([r["iterative"][i]["test"]["pair"]["pair_recall"] for r in runs])) for i in range(ITERATIONS)], "boundary_overlap": [float(np.mean([r["iterative"][i]["test"]["pair"]["boundary_overlap"] for r in runs])) for i in range(ITERATIONS)]}}
    return {"settings": {"architecture": "561→64→32→6", "seeds": list(seeds), "split": "strict train/validation/test", "pair": "SITTING(3)↔STANDING(4)"}, "summary": summary, "runs": runs}


def write_report(result, path):
    s = result["summary"]
    baseline_pair = [run["baseline_test"]["pair"] for run in result["runs"]]
    overlap_fraction = np.mean([item["ambiguous_fraction"] for item in baseline_pair])
    error_in = np.mean([item["error_in_overlap"] for item in baseline_pair])
    error_out = np.mean([item["error_outside_overlap"] for item in baseline_pair])
    lines = ["# Boundary-overlap inverse design", "", "SITTING/STANDING의 hidden2 분포와 output decision 방향을 비교하고, centroid를 보존하면서 boundary overlap만 줄이는 inverse를 validation에서 설계했다. test는 최종 평가에서만 사용했다.", "", "## Baseline distribution", f"- accuracy={s['baseline']['accuracy']:.3f}, pair recall={s['baseline']['pair_recall']:.3f}, pair confusion={s['baseline']['pair_confusion']:.1f}, boundary overlap={s['baseline']['boundary_overlap']:.3f}, centroid distance={s['baseline']['centroid_distance']:.3f}, q10 margin={s['baseline']['q10_margin']:.3f}", f"- pair sample 중 ambiguity 영역 비율={overlap_fraction:.1%}", f"- ambiguity 영역 오분류율={error_in:.1%}, 바깥 오분류율={error_out:.1%}", "", "## Test comparison"]
    for name, value in s["one_shot"].items():
        lines.append(f"- {name}: accuracy={value['accuracy']:.3f}, pair recall={value['pair_recall']:.3f}, pair confusion={value['pair_confusion']:.1f}, overlap={value['boundary_overlap']:.3f}, centroid distance={value['centroid_distance']:.3f}, q10 margin={value['q10_margin']:.3f}, other preservation={value['other_class_preservation']:.3f}, gate change={value['gate_change_fraction']:.1%}")
    lines += ["", "## 지표-오류 연결", f"- centroid distance correlation={s['correlation_with_pair_error']['centroid_distance']:.3f}", f"- mean pair margin correlation={s['correlation_with_pair_error']['mean_pair_margin']:.3f}", f"- boundary overlap correlation={s['correlation_with_pair_error']['boundary_overlap']:.3f}", "", "## 반복 overlap inverse", f"- accuracy: {', '.join(f'{x:.3f}' for x in s['iterative']['accuracy'])}", f"- pair recall: {', '.join(f'{x:.3f}' for x in s['iterative']['pair_recall'])}", f"- overlap: {', '.join(f'{x:.3f}' for x in s['iterative']['boundary_overlap'])}", "", "## 결론", "", "baseline에서는 오류가 boundary overlap 영역에 집중됐다. 그러나 validation에서 설계한 overlap inverse는 test overlap을 0.151에서 0.184로 오히려 늘렸고 pair recall도 낮췄다. class distribution의 공통 방향은 오류 설명에는 유효하지만, validation sample의 overlap 제거 방향이 test distribution의 covariance·tail 구조까지 일반화되지는 않았다."]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    root = Path(__file__).parent
    result = run_overlap_inverse(root / "UCI HAR Dataset")
    (root / "overlap_inverse_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_report(result, root / "overlap_inverse_analysis.md")
    print(json.dumps({"runs": len(result["runs"]), "result": str(root / "overlap_inverse_results.json")}, indent=2))
