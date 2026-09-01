"""Direct probability-mass overlap inverse design for SITTING/STANDING."""

import json
from pathlib import Path

import numpy as np

from .class_pair_inverse_experiment import _centroid_jacobian
from .distribution_inverse_experiment import VARIANCE_FACTOR, _distribution_constraints
from .generalization_experiment import _copy, _forward, _init_model
from .inverse_geometry_experiment import ITERATIONS, _apply_hidden_delta, _gate_state, _hidden_vector, _inverse_delta, _same_norm_delta, _train_model
from .margin_inverse_experiment import _geometry, _margin_metrics
from .overlap_inverse_experiment import FIRST, SECOND, _boundary, _centroid_constraints
from .validation_selective_inverse_experiment import _strict_split


OBJECTIVE_FACTOR = 0.7
STEP_FRACTION = 0.2
POC_SEEDS = (7, 11, 19)


def _sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def _direct_state(model, inputs, targets):
    h2 = _forward(model, inputs, 2)[0][-1]
    logits = h2 @ model["w2"] + model["b2"]
    score = logits[:, FIRST] - logits[:, SECOND]
    signed = np.where(targets == FIRST, score, -score)
    probability_mass = _sigmoid(-signed)
    pair_mask = np.isin(targets, (FIRST, SECOND))
    return h2, logits, score, signed, probability_mass, pair_mask


def _direct_metrics(model, inputs, targets, baseline_predictions=None):
    h2, logits, score, signed, wrong_probability, pair_mask = _direct_state(model, inputs, targets)
    predictions = logits.argmax(axis=1)
    pair_confusion = int(np.sum((targets == FIRST) & (predictions == SECOND)) + np.sum((targets == SECOND) & (predictions == FIRST)))
    result = {
        "wrong_probability_mass": float(wrong_probability[pair_mask].mean()),
        "wrong_side_mass": float(np.mean(signed[pair_mask] < 0.0)),
        "pair_recall": float(np.mean(predictions[pair_mask] == targets[pair_mask])),
        "pair_confusion": pair_confusion,
        "mean_signed_score": float(signed[pair_mask].mean()),
    }
    if baseline_predictions is not None:
        result["other_class_preservation"] = float(np.mean(predictions[~pair_mask] == baseline_predictions[~pair_mask]))
    return result


def _weighted_score_row(model, inputs, weights, direction):
    z0 = inputs @ model["w0"] + model["b0"]
    h0 = np.maximum(z0, 0.0)
    z1 = h0 @ model["w1"] + model["b1"]
    gate0, gate1 = z0 > 0, z1 > 0
    row = np.zeros(model["w0"].size + model["b0"].size + model["w1"].size + model["b1"].size)
    for output, coefficient in enumerate(direction):
        weighted = weights * gate1[:, output]
        cross = inputs.T @ (weighted[:, None] * gate0)
        w0_row = (cross * model["w1"][:, output][None, :]).ravel()
        b0_row = (weighted[:, None] * gate0).sum(axis=0) * model["w1"][:, output]
        w1_block = np.zeros_like(model["w1"])
        w1_block[:, output] = (weighted[:, None] * h0).sum(axis=0)
        b1_row = np.zeros_like(model["b1"])
        b1_row[output] = weighted.sum()
        row += coefficient * np.concatenate([w0_row, b0_row, w1_block.ravel(), b1_row])
    return row


def _direct_constraints(model, inputs, targets):
    h2, logits, score, signed, wrong_probability, pair_mask = _direct_state(model, inputs, targets)
    direction = _boundary(model)
    # d wrong_probability / d signed_score = -p(1-p), with signed score positive for the true class.
    weights = np.zeros(len(inputs))
    scale = wrong_probability * (1.0 - wrong_probability) / max(np.sum(pair_mask), 1)
    weights[targets == FIRST] = -scale[targets == FIRST]
    weights[targets == SECOND] = scale[targets == SECOND]
    objective_row = _weighted_score_row(model, inputs, weights, direction)
    objective = float(wrong_probability[pair_mask].mean())
    rows, gains = [objective_row], [(OBJECTIVE_FACTOR - 1.0) * objective]
    # Preserve both pair centroids exactly in hidden2 and preserve other class centroid margins.
    blocks = _centroid_jacobian(model, inputs, targets)
    for label in (FIRST, SECOND):
        rows.extend(list(blocks[label]))
        gains.extend([0.0] * 32)
    centroids = np.asarray([h2[targets == label].mean(axis=0) for label in range(6)])
    centroid_logits = centroids @ model["w2"] + model["b2"]
    for label in range(6):
        if label in (FIRST, SECOND):
            continue
        order = np.argsort(centroid_logits[label])[::-1]
        competitor = int(order[0] if order[0] != label else order[1])
        rows.append(np.einsum("j,jp->p", model["w2"][:, label] - model["w2"][:, competitor], blocks[label]))
        gains.append(0.0)
    return np.asarray(rows), np.asarray(gains), {"objective": objective, "pair_count": int(np.sum(pair_mask)), "centroid_constraints": 64}


def _evaluate(model, inputs, targets, baseline_predictions=None):
    result = {"margin": _margin_metrics(model, inputs, targets), "direct": _direct_metrics(model, inputs, targets, baseline_predictions), "geometry": _geometry(model, inputs, targets)}
    return result


def _run_trained_seed(data, seed):
    model = _train_model(_init_model(561, (64, 32), 6, seed), data["train_x"], data["train_y"], seed)
    validation_baseline = _evaluate(model, data["val_x"], data["val_y"])
    direct_j, direct_target, direct_info = _direct_constraints(model, data["val_x"], data["val_y"])
    direct_delta = _inverse_delta(direct_j, direct_target)
    variance_j, variance_target = _distribution_constraints(model, data["val_x"], data["val_y"])
    variance_delta = _inverse_delta(variance_j, variance_target)
    centroid_j, centroid_target = _centroid_constraints(model, data["val_x"], data["val_y"])
    centroid_delta = _inverse_delta(centroid_j, centroid_target)
    variance_delta *= np.linalg.norm(direct_delta) / max(np.linalg.norm(variance_delta), 1e-12)
    centroid_delta *= np.linalg.norm(direct_delta) / max(np.linalg.norm(centroid_delta), 1e-12)
    rng = np.random.default_rng(seed + 9600)
    candidates = {"direct_overlap_inverse": _apply_hidden_delta(model, direct_delta), "variance_reduction_inverse": _apply_hidden_delta(model, variance_delta), "centroid_inverse": _apply_hidden_delta(model, centroid_delta), "random_same_norm": _apply_hidden_delta(model, _same_norm_delta(direct_delta, rng))}
    baseline_val_predictions = _forward(model, data["val_x"], 2)[2].argmax(axis=1)
    candidate_results = {}
    for name, candidate in candidates.items():
        candidate_results[name] = {"validation": _evaluate(candidate, data["val_x"], data["val_y"], baseline_val_predictions), "delta_norm": float(np.linalg.norm(_hidden_vector(candidate) - _hidden_vector(model))), "gate_change_fraction": float(np.mean(_gate_state(candidate, data["val_x"]) != _gate_state(model, data["val_x"]))), "model": candidate}
    current = _copy(model)
    iterative = []
    for iteration in range(ITERATIONS):
        jacobian, target, _ = _direct_constraints(current, data["val_x"], data["val_y"])
        step = _inverse_delta(jacobian, target)
        current = _apply_hidden_delta(current, step, STEP_FRACTION)
        iterative.append({"iteration": iteration + 1, "validation": _evaluate(current, data["val_x"], data["val_y"], baseline_val_predictions), "step_norm": float(np.linalg.norm(step) * STEP_FRACTION), "model": _copy(current)})
    test_predictions = _forward(model, data["test_x"], 2)[2].argmax(axis=1)
    baseline_test = _evaluate(model, data["test_x"], data["test_y"], test_predictions)
    for result in candidate_results.values():
        candidate = result.pop("model")
        result["test"] = _evaluate(candidate, data["test_x"], data["test_y"], test_predictions)
    for result in iterative:
        candidate = result.pop("model")
        result["test"] = _evaluate(candidate, data["test_x"], data["test_y"], test_predictions)
    return {"seed": seed, "validation_baseline": validation_baseline, "baseline_test": baseline_test, "direct_info": direct_info, "direct_jacobian_shape": list(direct_j.shape), "one_shot": candidate_results, "iterative": iterative}


def _summary(runs):
    names = ("direct_overlap_inverse", "variance_reduction_inverse", "centroid_inverse", "random_same_norm")
    def metrics(name):
        rows = [r["one_shot"][name]["test"] for r in runs]
        return {"wrong_probability_mass": float(np.mean([x["direct"]["wrong_probability_mass"] for x in rows])), "wrong_side_mass": float(np.mean([x["direct"]["wrong_side_mass"] for x in rows])), "pair_recall": float(np.mean([x["direct"]["pair_recall"] for x in rows])), "pair_confusion": float(np.mean([x["direct"]["pair_confusion"] for x in rows])), "accuracy": float(np.mean([x["margin"]["accuracy"] for x in rows])), "q10_margin": float(np.mean([x["margin"]["q10_margin"] for x in rows])), "other_class_preservation": float(np.mean([x["direct"].get("other_class_preservation", 1.0) for x in rows]))}
    return {"baseline": {"wrong_probability_mass": float(np.mean([r["baseline_test"]["direct"]["wrong_probability_mass"] for r in runs])), "wrong_side_mass": float(np.mean([r["baseline_test"]["direct"]["wrong_side_mass"] for r in runs])), "pair_recall": float(np.mean([r["baseline_test"]["direct"]["pair_recall"] for r in runs])), "pair_confusion": float(np.mean([r["baseline_test"]["direct"]["pair_confusion"] for r in runs])), "accuracy": float(np.mean([r["baseline_test"]["margin"]["accuracy"] for r in runs])), "q10_margin": float(np.mean([r["baseline_test"]["margin"]["q10_margin"] for r in runs]))}, "one_shot": {name: metrics(name) for name in names}, "iterative": {"accuracy": [float(np.mean([r["iterative"][i]["test"]["margin"]["accuracy"] for r in runs])) for i in range(ITERATIONS)], "wrong_probability_mass": [float(np.mean([r["iterative"][i]["test"]["direct"]["wrong_probability_mass"] for r in runs])) for i in range(ITERATIONS)], "pair_recall": [float(np.mean([r["iterative"][i]["test"]["direct"]["pair_recall"] for r in runs])) for i in range(ITERATIONS)]}}


def _run_random_init_poc(data_dir):
    data = _strict_split(data_dir)
    rows = []
    for seed in POC_SEEDS:
        model = _init_model(561, (64, 32), 6, seed)
        baseline = _evaluate(model, data["test_x"], data["test_y"])
        current = model
        for _ in range(ITERATIONS):
            jacobian, target, _ = _direct_constraints(current, data["val_x"], data["val_y"])
            current = _apply_hidden_delta(current, _inverse_delta(jacobian, target), STEP_FRACTION)
        final = _evaluate(current, data["test_x"], data["test_y"])
        rows.append({"seed": seed, "baseline": baseline, "final": final})
    return {"seeds": list(POC_SEEDS), "rows": rows, "baseline_accuracy": float(np.mean([x["baseline"]["margin"]["accuracy"] for x in rows])), "final_accuracy": float(np.mean([x["final"]["margin"]["accuracy"] for x in rows])), "baseline_pair_recall": float(np.mean([x["baseline"]["direct"]["pair_recall"] for x in rows])), "final_pair_recall": float(np.mean([x["final"]["direct"]["pair_recall"] for x in rows]))}


def run_direct_overlap(data_dir, seeds=(7, 11, 19, 23, 31)):
    data = _strict_split(data_dir)
    runs = [_run_trained_seed(data, seed) for seed in seeds]
    result = {"settings": {"architecture": "561→64→32→6", "seeds": list(seeds), "split": "strict train/validation/test", "objective_factor": OBJECTIVE_FACTOR}, "summary": _summary(runs), "runs": runs}
    val_base = np.mean([r["validation_baseline"]["direct"]["wrong_probability_mass"] for r in runs])
    val_direct = np.mean([r["one_shot"]["direct_overlap_inverse"]["validation"]["direct"]["wrong_probability_mass"] for r in runs])
    test_base = result["summary"]["baseline"]["wrong_probability_mass"]
    test_direct = result["summary"]["one_shot"]["direct_overlap_inverse"]["wrong_probability_mass"]
    if val_direct < val_base and test_direct < test_base:
        result["random_initialization_poc"] = _run_random_init_poc(data_dir)
    else:
        result["random_initialization_poc"] = {"status": "skipped", "reason": "direct overlap did not decrease in both validation and unseen test"}
    return result


def write_report(result, path):
    s = result["summary"]
    val_base = np.mean([r["validation_baseline"]["direct"]["wrong_probability_mass"] for r in result["runs"]])
    val_direct = np.mean([r["one_shot"]["direct_overlap_inverse"]["validation"]["direct"]["wrong_probability_mass"] for r in result["runs"]])
    lines = ["# Direct probability-mass overlap inverse", "", "validation의 평균 wrong-class probability mass 자체를 objective로 삼고, pair centroid와 다른 class centroid margin을 보존했다. test는 모든 update 이후 최종 평가에만 사용했다.", "", f"validation direct objective: {val_base:.3f} → {val_direct:.3f}", "", "## Test 결과"]
    for name, value in {"baseline": s["baseline"], **s["one_shot"]}.items():
        lines.append(f"- {name}: wrong probability mass={value['wrong_probability_mass']:.3f}, wrong-side mass={value['wrong_side_mass']:.3f}, pair recall={value['pair_recall']:.3f}, confusion={value['pair_confusion']:.1f}, accuracy={value['accuracy']:.3f}, q10 margin={value['q10_margin']:.3f}, other preservation={value.get('other_class_preservation', 1.0):.3f}")
    lines += ["", "## 반복 direct inverse", f"- accuracy: {', '.join(f'{x:.3f}' for x in s['iterative']['accuracy'])}", f"- wrong probability mass: {', '.join(f'{x:.3f}' for x in s['iterative']['wrong_probability_mass'])}", f"- pair recall: {', '.join(f'{x:.3f}' for x in s['iterative']['pair_recall'])}", "", "## Random initialization PoC", f"`{json.dumps(result['random_initialization_poc'], ensure_ascii=False)}`", "", "## 결론", "", "A를 probability mass 기준으로는 약하게 만족했지만, pair recall과 전체 accuracy는 개선되지 않았다. 따라서 direct objective의 연속값 감소가 곧 discrete classification 오류 감소를 의미하지 않는다. 반복 direct step에서는 test objective가 다시 증가해 일반화가 안정적이지 않았다. random initialization PoC도 pair recall 일부 변화만 만들었고 전체 정확도는 약 0.161에서 0.154로 낮아져 의미 있는 6-class classifier를 만들지 못했다."]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    root = Path(__file__).parent
    result = run_direct_overlap(root / "UCI HAR Dataset")
    (root / "direct_overlap_inverse_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_report(result, root / "direct_overlap_inverse_analysis.md")
    print(json.dumps({"runs": len(result["runs"]), "result": str(root / "direct_overlap_inverse_results.json")}, indent=2))
