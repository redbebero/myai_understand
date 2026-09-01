"""Test W^T(p-y) against hidden representation movement."""

import json
from pathlib import Path

import numpy as np

from .gradient_geometry_experiment import _copy_model, _cosine, _loss
from .geometry_principle_experiment import _accuracy_from_h2, _class_subspace, _remove_subspace, geometry_metrics
from .uci_har_experiment import _adam_update, baseline_forward, load_data, new_model, softmax


def _pearson(first, second):
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    first -= first.mean()
    second -= second.mean()
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    return float(first @ second / denominator) if denominator else 0.0


def _pairwise_cosines(vectors):
    vectors = np.asarray(vectors, dtype=float)
    vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    values = []
    for first in range(len(vectors)):
        for second in range(first + 1, len(vectors)):
            values.append(float(vectors[first] @ vectors[second]))
    return values


def _gradients(model, batch_x, batch_y):
    h1, h2, probabilities = baseline_forward(model, batch_x)
    error = probabilities.copy()
    error[np.arange(len(batch_y)), batch_y] -= 1.0
    error /= len(batch_y)
    gradient_h = error @ model["w3"].T
    dh2 = gradient_h * (h2 > 0)
    gradients = {"w3": h2.T @ error, "b3": error.sum(axis=0), "w2": h1.T @ dh2, "b2": dh2.sum(axis=0)}
    dh1 = (dh2 @ model["w2"].T) * (h1 > 0)
    gradients.update({"w1": batch_x.T @ dh1, "b1": dh1.sum(axis=0)})
    return h2, probabilities, gradients, gradient_h


def _direction_batch_metrics(before_h2, after_h2, probabilities, targets, negative_gradient, weight_vectors):
    target_gradient, target_delta = [], []
    wrong_prob, wrong_gradient, wrong_delta = [], [], []
    for index, target in enumerate(targets):
        target_weight = weight_vectors[target]
        delta = after_h2[index] - before_h2[index]
        target_gradient.append(_cosine(target_weight, negative_gradient[index]))
        target_delta.append(_cosine(target_weight, delta))
        wrong = int(np.argsort(probabilities[index])[-2] if np.argmax(probabilities[index]) == target else np.argmax(probabilities[index]))
        wrong_prob.append(float(probabilities[index, wrong]))
        wrong_gradient.append(_cosine(weight_vectors[wrong], negative_gradient[index]))
        wrong_delta.append(_cosine(weight_vectors[wrong], delta))
    wrong_prob = np.asarray(wrong_prob)
    high = wrong_prob >= np.quantile(wrong_prob, 0.75)
    low = wrong_prob <= np.quantile(wrong_prob, 0.25)
    return {
        "target_weight_cosine_negative_gradient": float(np.mean(target_gradient)),
        "target_weight_cosine_actual_delta": float(np.mean(target_delta)),
        "wrong_probability": float(np.mean(wrong_prob)),
        "wrong_weight_cosine_negative_gradient": float(np.mean(wrong_gradient)),
        "wrong_weight_cosine_actual_delta": float(np.mean(wrong_delta)),
        "wrong_probability_gradient_cosine_correlation": _pearson(wrong_prob, wrong_gradient),
        "wrong_probability_delta_cosine_correlation": _pearson(wrong_prob, wrong_delta),
        "high_wrong_probability_gradient_cosine": float(np.mean(np.asarray(wrong_gradient)[high])),
        "low_wrong_probability_gradient_cosine": float(np.mean(np.asarray(wrong_gradient)[low])),
    }


def _centroid_metrics(model, data, train_probe, test_probe):
    test_h2 = baseline_forward(model, data["test_x"][test_probe])[1]
    targets = data["test_y"][test_probe]
    centroids = np.array([test_h2[targets == label].mean(axis=0) for label in range(6)])
    weights = model["w3"].T
    return {
        "centroid_weight_cosines": [_cosine(centroids[label], weights[label]) for label in range(6)],
        "mean_centroid_weight_cosine": float(np.mean([_cosine(centroids[label], weights[label]) for label in range(6)])),
        "weight_pair_cosines": _pairwise_cosines(weights),
        "centroid_pair_cosines": _pairwise_cosines(centroids),
        "weight_centroid_pair_correlation": _pearson(_pairwise_cosines(weights), _pairwise_cosines(centroids)),
    }


def trace_seed(data, seed, batch_size=128):
    model = new_model(seed)
    rng = np.random.default_rng(seed + 1)
    moments = {name: [np.zeros_like(value), np.zeros_like(value)] for name, value in model.items()}
    train_probe = np.arange(min(600, len(data["train_y"])))
    test_probe = np.random.default_rng(0).choice(len(data["test_y"]), size=600, replace=False)
    records = []
    for update, batch_indices in enumerate(np.array_split(rng.permutation(len(data["train_x"])), max(1, len(data["train_x"]) // batch_size))):
        batch_x, batch_y = data["train_x"][batch_indices], data["train_y"][batch_indices]
        before_h2, before_probabilities, gradients, gradient_h = _gradients(model, batch_x, batch_y)
        old_model = _copy_model(model)
        _adam_update(model, gradients, moments, update + 1, 0.001)
        after_h2, after_probabilities = baseline_forward(model, batch_x)[1:]
        weights_before = old_model["w3"].T
        direction = _direction_batch_metrics(before_h2, after_h2, before_probabilities, batch_y, -gradient_h, weights_before)
        records.append({
            "update": update + 1,
            "loss_before": _loss(before_probabilities, batch_y),
            "loss_after": _loss(after_probabilities, batch_y),
            "target_class_direction": direction,
            "weight_update_cosines": {name: _cosine(-gradients[name], model[name] - old_model[name]) for name in ("w1", "w2", "w3")},
            "centroid_geometry": _centroid_metrics(model, data, train_probe, test_probe),
        })
    return records


def run_crossentropy_geometry(data_dir, seeds=(7, 11, 19, 23, 31)):
    data = load_data(data_dir)
    runs = [{"seed": seed, "records": trace_seed(data, seed)} for seed in seeds]
    stages = [
        {"prediction": "-∂L/∂h = W^T(y-p)는 정답 class weight 방향 성분을 가진다.", "measurement": "정답 weight와 negative gradient/actual Δh의 cosine을 batch마다 측정한다."},
        {"prediction": "오답 class 확률이 높을수록 해당 weight 방향의 음의 성분이 강해진다.", "measurement": "wrong probability와 wrong-weight cosine의 상관 및 상·하위 quartile을 비교한다."},
        {"prediction": "학습이 진행되면 class centroid가 자기 output weight와 정렬된다.", "measurement": "checkpoint별 centroid-weight cosine과 weight/centroid pair-angle 구조를 추적한다."},
        {"prediction": "weight geometry와 centroid geometry가 함께 형성된다.", "measurement": "두 pairwise cosine 행렬의 상관과 seed 간 반복성을 비교한다."},
    ]
    direction_records = [record["target_class_direction"] for run in runs for record in run["records"]]
    stages[0]["actual_result"] = {"target_weight_vs_negative_gradient": float(np.mean([row["target_weight_cosine_negative_gradient"] for row in direction_records])), "target_weight_vs_actual_delta": float(np.mean([row["target_weight_cosine_actual_delta"] for row in direction_records]))}
    stages[1]["actual_result"] = {"wrong_weight_vs_negative_gradient": float(np.mean([row["wrong_weight_cosine_negative_gradient"] for row in direction_records])), "wrong_probability_gradient_correlation": float(np.mean([row["wrong_probability_gradient_cosine_correlation"] for row in direction_records])), "high_vs_low_wrong_probability_cosine": [float(np.mean([row["high_wrong_probability_gradient_cosine"] for row in direction_records])), float(np.mean([row["low_wrong_probability_gradient_cosine"] for row in direction_records]))]}
    stages[2]["actual_result"] = {"centroid_weight_cosine_initial": float(np.mean([run["records"][0]["centroid_geometry"]["mean_centroid_weight_cosine"] for run in runs])), "centroid_weight_cosine_final": float(np.mean([run["records"][-1]["centroid_geometry"]["mean_centroid_weight_cosine"] for run in runs]))}
    stages[3]["actual_result"] = {"weight_centroid_pair_correlation_initial": float(np.mean([run["records"][0]["centroid_geometry"]["weight_centroid_pair_correlation"] for run in runs])), "weight_centroid_pair_correlation_final": float(np.mean([run["records"][-1]["centroid_geometry"]["weight_centroid_pair_correlation"] for run in runs])), "mean_abs_weight_pair_cosine_change": float(np.mean([np.mean(np.abs(np.asarray(run["records"][-1]["centroid_geometry"]["weight_pair_cosines"]) - np.asarray(run["records"][0]["centroid_geometry"]["weight_pair_cosines"]))) for run in runs])), "mean_abs_centroid_pair_cosine_change": float(np.mean([np.mean(np.abs(np.asarray(run["records"][-1]["centroid_geometry"]["centroid_pair_cosines"]) - np.asarray(run["records"][0]["centroid_geometry"]["centroid_pair_cosines"]))) for run in runs]))}
    return {"settings": {"seeds": list(seeds), "updates_per_seed": len(runs[0]["records"])}, "runs": runs, "stages": stages}


def write_report(result, path):
    lines = ["# Cross-entropy gradient와 hidden geometry", ""]
    for index, stage in enumerate(result["stages"], 1):
        lines += [f"## {index}", "", f"- 수식의 예측: {stage['prediction']}", f"- 실제 측정: {stage['measurement']}", f"- 실제 결과: {stage['actual_result']}", "- 맞지 않는 점: batch별 원자료에서 cosine과 정렬이 완전하지 않은 부분을 확인한다.", "- 수정된 메커니즘: `cross-entropy error → output weights → hidden movement → geometry → boundary`의 어느 단계가 약한지 구분한다.", ""]
    lines += ["## 최소 메커니즘", "", "`W^T(p-y) → Adam weight update → Δh → class centroid/weight geometry → decision boundary`"]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    root = Path(__file__).parent
    result = run_crossentropy_geometry(root / "UCI HAR Dataset")
    (root / "crossentropy_geometry_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_report(result, root / "crossentropy_geometry_analysis.md")
    print(json.dumps({"runs": len(result["runs"]), "result": str(root / "crossentropy_geometry_results.json")}, indent=2))
