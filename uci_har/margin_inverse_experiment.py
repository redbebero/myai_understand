"""Reverse-design hidden geometry aligned with the classifier margin."""

import json
from pathlib import Path

import numpy as np

from .generalization_experiment import _adam_update, _copy, _forward, _gradients, _init_model
from .inverse_geometry_experiment import (
    ITERATIONS,
    TRAIN_EPOCHS,
    _accuracy,
    _apply_hidden_delta,
    _centroid_jacobian as _dynamic_centroid_jacobian,
    _centroids as _dynamic_centroids,
    _gate_state,
    _hidden_vector,
    _inverse_delta,
    _same_norm_delta,
    _target_geometry as _dynamic_target_geometry,
    _train_model,
)
from .uci_har_experiment import load_data


CLASS_COUNT = 6
SAMPLE_GAIN = 1.0
CENTROID_GAIN = 0.5
STEP_FRACTION = 0.2


def _hidden_rows(model, inputs):
    z0 = inputs @ model["w0"] + model["b0"]
    h0 = np.maximum(z0, 0.0)
    z1 = h0 @ model["w1"] + model["b1"]
    gate0, gate1 = z0 > 0, z1 > 0
    rows = []
    for sample in range(len(inputs)):
        for output in range(model["w1"].shape[1]):
            w0_row = np.outer(inputs[sample], gate0[sample] * gate1[sample, output] * model["w1"][:, output]).ravel()
            b0_row = gate0[sample] * gate1[sample, output] * model["w1"][:, output]
            w1_block = np.zeros_like(model["w1"])
            w1_block[:, output] = h0[sample] * gate1[sample, output]
            b1_row = np.zeros_like(model["b1"])
            b1_row[output] = float(gate1[sample, output])
            rows.append(np.concatenate([w0_row, b0_row, w1_block.ravel(), b1_row]))
    return np.asarray(rows)


def _centroids(model, inputs, targets, labels=range(CLASS_COUNT)):
    h2 = _forward(model, inputs, 2)[0][-1]
    return np.array([h2[targets == label].mean(axis=0) for label in labels])


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
            rows.append(np.concatenate([w0_row, b0_row, w1_block.ravel(), [gate1[:, output].mean()]]))
    return np.asarray(rows)


def _margin_state(model, inputs, targets):
    h2, _, probabilities = _forward(model, inputs, 2)
    logits = h2[-1] @ model["w2"] + model["b2"]
    return _margin_from_logits(h2[-1], logits, targets)


def _margin_from_logits(h2, logits, targets):
    margins, wrong = [], []
    for row, label in zip(logits, targets):
        order = np.argsort(row)[::-1]
        competitor = order[0] if order[0] != label else order[1]
        margins.append(float(row[label] - row[competitor]))
        wrong.append(int(competitor))
    return np.asarray(margins), np.asarray(wrong), h2, logits


def _margin_metrics(model, inputs, targets):
    margins, _, _, logits = _margin_state(model, inputs, targets)
    prediction = logits.argmax(axis=1)
    confusion = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=int)
    for target, predicted in zip(targets, prediction):
        confusion[int(target), int(predicted)] += 1
    return {
        "accuracy": float(np.mean(prediction == targets)),
        "mean_margin": float(margins.mean()),
        "min_margin": float(margins.min()),
        "q10_margin": float(np.quantile(margins, 0.1)),
        "confusion": confusion.tolist(),
    }


def _geometry(model, inputs, targets):
    h2 = _forward(model, inputs, 2)[0][-1]
    labels = np.unique(targets)
    centroids = _centroids(model, inputs, targets, labels)
    within = np.mean([np.mean((h2[targets == label] - centroids[i]) ** 2) for i, label in enumerate(labels)])
    center = centroids.mean(axis=0)
    between = np.mean((centroids - center) ** 2)
    nearest = labels[np.linalg.norm(h2[:, None, :] - centroids[None, :, :], axis=2).argmin(axis=1)]
    return {"within": float(within), "between": float(between), "separation_ratio": float(between / max(within, 1e-12)), "nearest_centroid_accuracy": float(np.mean(nearest == targets)), "mean_pair_distance": float(np.mean(np.linalg.norm(centroids[:, None] - centroids[None, :], axis=2)[np.triu_indices(len(labels), 1)]))}


def _margin_target(model, inputs, targets):
    margins, wrong, h2, _ = _margin_state(model, inputs, targets)
    selected = []
    for label in range(CLASS_COUNT):
        candidates = np.flatnonzero(targets == label)
        selected.append(int(candidates[np.argmin(margins[candidates])]))
    selected = np.asarray(selected)
    directions, target_gains, descriptions = [], [], []
    for index, gain in [(index, SAMPLE_GAIN) for index in selected]:
        label, competitor = int(targets[index]), int(wrong[index])
        direction = model["w2"][:, label] - model["w2"][:, competitor]
        directions.append(direction)
        target_gains.append(gain)
        descriptions.append({"kind": "low_margin_sample", "index": int(index), "label": label, "competitor": competitor, "margin": float(margins[index]), "gain": gain})
    centroids = _centroids(model, inputs, targets)
    centroid_logits = centroids @ model["w2"] + model["b2"]
    centroid_margins, centroid_wrong, _, _ = _margin_from_logits(centroids, centroid_logits, np.arange(CLASS_COUNT))
    for label in range(CLASS_COUNT):
        competitor = int(centroid_wrong[label])
        direction = model["w2"][:, label] - model["w2"][:, competitor]
        directions.append(direction)
        target_gains.append(CENTROID_GAIN)
        descriptions.append({"kind": "centroid", "label": label, "competitor": competitor, "margin": float(centroid_margins[label]), "gain": CENTROID_GAIN})
    return selected, np.asarray(directions), np.asarray(target_gains), descriptions


def _design_jacobian(model, inputs, targets, selected, directions):
    sample_rows = _hidden_rows(model, inputs[selected]).reshape(len(selected), 32, -1)
    centroid_rows = _centroid_jacobian(model, inputs, targets).reshape(CLASS_COUNT, 32, -1)
    rows = np.concatenate([sample_rows, centroid_rows])
    return np.einsum("ij,ijp->ip", directions, rows)


def _dynamic_centroid_inverse(model, inputs, targets, factor=1.2):
    selected = targets < 3
    centroids = _dynamic_centroids(model, inputs[selected], targets[selected])
    target_delta = (_dynamic_target_geometry(centroids) - centroids).ravel()
    jacobian = _dynamic_centroid_jacobian(model, inputs, targets)
    return _inverse_delta(jacobian, target_delta)


def _evaluate(model, data):
    return {"margin": _margin_metrics(model, data["test_x"], data["test_y"]), "geometry": _geometry(model, data["test_x"], data["test_y"]), "dynamic_geometry": _geometry(model, data["test_x"][data["test_y"] < 3], data["test_y"][data["test_y"] < 3])}


def run_seed(data, seed):
    model = _train_model(_init_model(data["train_x"].shape[1], (64, 32), CLASS_COUNT, seed), data["train_x"], data["train_y"], seed)
    baseline = _evaluate(model, data)
    selected, directions, margin_target_delta, descriptions = _margin_target(model, data["train_x"], data["train_y"])
    margin_jacobian = _design_jacobian(model, data["train_x"], data["train_y"], selected, directions)
    margin_inverse = _inverse_delta(margin_jacobian, margin_target_delta)
    centroid_inverse = _dynamic_centroid_inverse(model, data["train_x"], data["train_y"])
    gradient = _gradients(model, data["train_x"], data["train_y"], 2)[3]
    gradient_vector = np.concatenate([gradient["w0"].ravel(), gradient["b0"], gradient["w1"].ravel(), gradient["b1"]])
    rng = np.random.default_rng(seed + 9100)
    norm = np.linalg.norm(margin_inverse)
    candidates = {
        "margin_inverse": _apply_hidden_delta(model, margin_inverse),
        "centroid_inverse": _apply_hidden_delta(model, centroid_inverse),
        "random_same_norm": _apply_hidden_delta(model, _same_norm_delta(margin_inverse, rng)),
        "gradient_same_norm": _apply_hidden_delta(model, -gradient_vector * (norm / max(np.linalg.norm(gradient_vector), 1e-12))),
    }
    results = {}
    for name, candidate in candidates.items():
        metric = _evaluate(candidate, data)
        metric["delta_norm"] = float(np.linalg.norm(_hidden_vector(candidate) - _hidden_vector(model)))
        metric["gate_change_fraction"] = float(np.mean(_gate_state(candidate, data["train_x"]) != _gate_state(model, data["train_x"])))
        results[name] = metric
    iterative = []
    current = _copy(model)
    for iteration in range(ITERATIONS):
        selected_now, directions_now, target_now, _ = _margin_target(current, data["train_x"], data["train_y"])
        step = _inverse_delta(_design_jacobian(current, data["train_x"], data["train_y"], selected_now, directions_now), target_now)
        current = _apply_hidden_delta(current, step, STEP_FRACTION)
        metric = _evaluate(current, data)
        metric.update({"iteration": iteration + 1, "step_norm": float(np.linalg.norm(step) * STEP_FRACTION), "gate_change_fraction": float(np.mean(_gate_state(current, data["train_x"]) != _gate_state(model, data["train_x"])))})
        iterative.append(metric)
    return {"seed": seed, "target_descriptions": descriptions, "jacobian_shape": list(margin_jacobian.shape), "baseline": baseline, "one_shot": results, "iterative": iterative}


def run_margin_inverse(data_dir, seeds=(7, 11, 19, 23, 31)):
    data = load_data(data_dir)
    runs = [run_seed(data, seed) for seed in seeds]
    names = ("margin_inverse", "centroid_inverse", "random_same_norm", "gradient_same_norm")
    summary = {
        "baseline": {"accuracy": float(np.mean([r["baseline"]["margin"]["accuracy"] for r in runs])), "mean_margin": float(np.mean([r["baseline"]["margin"]["mean_margin"] for r in runs])), "q10_margin": float(np.mean([r["baseline"]["margin"]["q10_margin"] for r in runs])), "min_margin": float(np.mean([r["baseline"]["margin"]["min_margin"] for r in runs]))},
        "one_shot": {name: {"accuracy": float(np.mean([r["one_shot"][name]["margin"]["accuracy"] for r in runs])), "mean_margin": float(np.mean([r["one_shot"][name]["margin"]["mean_margin"] for r in runs])), "q10_margin": float(np.mean([r["one_shot"][name]["margin"]["q10_margin"] for r in runs])), "min_margin": float(np.mean([r["one_shot"][name]["margin"]["min_margin"] for r in runs])), "separation_ratio": float(np.mean([r["one_shot"][name]["geometry"]["separation_ratio"] for r in runs])), "gate_change_fraction": float(np.mean([r["one_shot"][name]["gate_change_fraction"] for r in runs]))} for name in names},
        "iterative": {"accuracy": [float(np.mean([r["iterative"][i]["margin"]["accuracy"] for r in runs])) for i in range(ITERATIONS)], "mean_margin": [float(np.mean([r["iterative"][i]["margin"]["mean_margin"] for r in runs])) for i in range(ITERATIONS)], "q10_margin": [float(np.mean([r["iterative"][i]["margin"]["q10_margin"] for r in runs])) for i in range(ITERATIONS)]},
    }
    return {"settings": {"architecture": "561→64→32→6", "seeds": list(seeds), "train_epochs": TRAIN_EPOCHS, "sample_gain": SAMPLE_GAIN, "centroid_gain": CENTROID_GAIN, "iterations": ITERATIONS}, "summary": summary, "runs": runs}


def write_report(result, path):
    s = result["summary"]
    lines = ["# Decision-boundary aligned hidden geometry inverse design", "", "기존 UCI HAR 561→64→32→6 MLP를 seed별 80 epoch 학습하고, 각 class의 최저-margin sample과 class centroid를 정답 class의 output-weight 방향 안쪽으로 이동시키는 목표를 정의했다.", "", "## 결과", "", f"- baseline: accuracy={s['baseline']['accuracy']:.3f}, mean margin={s['baseline']['mean_margin']:.3f}, q10={s['baseline']['q10_margin']:.3f}, min={s['baseline']['min_margin']:.3f}"]
    for name, value in s["one_shot"].items():
        lines.append(f"- {name}: accuracy={value['accuracy']:.3f}, mean margin={value['mean_margin']:.3f}, q10={value['q10_margin']:.3f}, min={value['min_margin']:.3f}, separation ratio={value['separation_ratio']:.3f}, gate change={value['gate_change_fraction']:.1%}")
    lines += ["", "## 계산 흐름", "", "`decision boundary → margin deficit → Δh_target ∥ (w_y−w_j) → Jθ pseudoinverse → Δθ → actual margin/accuracy`", "", "margin inverse가 centroid inverse보다 분류 경계와 직접 정렬된 목표를 사용했는지, margin·accuracy·confusion의 seed별 원자료를 함께 비교해야 한다.", "", "## 반복 inverse", "", f"- accuracy: {', '.join(f'{x:.3f}' for x in s['iterative']['accuracy'])}", f"- mean margin: {', '.join(f'{x:.3f}' for x in s['iterative']['mean_margin'])}", f"- q10 margin: {', '.join(f'{x:.3f}' for x in s['iterative']['q10_margin'])}", "", "## 해석", "", "목표 margin이 실제로 증가하지만 accuracy가 증가하지 않으면, sample-level margin 목표와 전체 test distribution 또는 output-layer 정렬 사이에 불일치가 있다는 뜻이다. 반복 inverse의 개선·악화와 gate 변화는 local Jacobian 역산의 적용 범위를 보여준다."]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    root = Path(__file__).parent
    result = run_margin_inverse(root / "UCI HAR Dataset")
    (root / "margin_inverse_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_report(result, root / "margin_inverse_analysis.md")
    print(json.dumps({"runs": len(result["runs"]), "result": str(root / "margin_inverse_results.json")}, indent=2))
