"""Seed-invariant distributed representation analysis for the UCI HAR MLP."""

import json
from pathlib import Path

import numpy as np

from .interaction_experiment import evaluate_outputs
from .uci_har_experiment import CLASSES, HIDDEN, baseline_forward, load_data, train_baseline


DYNAMIC_LABELS = (0, 1, 2)
FEATURE_GROUP_PATTERNS = {
    "entropy": ("entropy",),
    "frequency": ("freq", "frequency"),
    "jerk": ("jerk",),
    "autocorrelation": ("arCoeff",),
}


def _pairwise_distances(values):
    values = np.asarray(values, dtype=float)
    squared = np.sum(values * values, axis=1, keepdims=True)
    distances = squared + squared.T - 2 * values @ values.T
    return np.sqrt(np.maximum(distances, 0.0))


def pattern_similarity(first, second):
    """Compare sample distance structure, invariant to hidden-unit permutation."""
    if len(first) != len(second):
        raise ValueError("patterns must contain the same samples")
    first_distances = _pairwise_distances(first)
    second_distances = _pairwise_distances(second)
    upper = np.triu_indices(len(first), 1)
    first_values = first_distances[upper]
    second_values = second_distances[upper]
    first_values -= first_values.mean()
    second_values -= second_values.mean()
    denominator = np.linalg.norm(first_values) * np.linalg.norm(second_values)
    if denominator == 0:
        return 1.0 if np.allclose(first_distances, second_distances) else 0.0
    return float(first_values @ second_values / denominator)


def class_activation_profile(activations, targets, labels=DYNAMIC_LABELS):
    activations = np.asarray(activations, dtype=float)
    centroids = {str(label): activations[targets == label].mean(axis=0) for label in labels}
    distances = {}
    for position, first in enumerate(labels):
        for second in labels[position + 1:]:
            distances[f"{first}-{second}"] = float(np.linalg.norm(centroids[str(first)] - centroids[str(second)]))
    active = activations > 0.0
    return {
        "sample_count": int(len(targets)),
        "centroids": centroids,
        "centroid_distances": distances,
        "mean_active_units": float(active.sum(axis=1).mean()),
        "active_rate": float(active.mean()),
        "by_class_active_units": {str(label): float(active[targets == label].sum(axis=1).mean()) for label in labels},
    }


def _class_contrast(activations, targets, labels=DYNAMIC_LABELS):
    means = [activations[targets == label].mean(axis=0) for label in labels]
    contrasts = np.zeros(activations.shape[1])
    for position, first in enumerate(means):
        for second in means[position + 1:]:
            contrasts = np.maximum(contrasts, np.abs(first - second))
    return contrasts


def activation_concentration(activations, targets, labels=DYNAMIC_LABELS):
    contrasts = _class_contrast(activations, targets, labels)
    energy = contrasts * contrasts
    total = float(energy.sum())
    order = np.argsort(energy)[::-1]
    fraction = lambda count: float(energy[order[:count]].sum() / total) if total else 0.0
    return {
        "contrast_energy": contrasts,
        "top_1_fraction": fraction(1),
        "top_2_fraction": fraction(2),
        "top_4_fraction": fraction(4),
        "top_8_fraction": fraction(8),
        "total_energy": total,
    }


def select_discriminative_units(activations, targets, top_k=4, labels=DYNAMIC_LABELS):
    contrasts = _class_contrast(activations, targets, labels)
    order = np.argsort(contrasts)[::-1]
    return {"units": [int(index) for index in order[:top_k]], "contrasts": contrasts}


def _pearson(first, second):
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    first -= first.mean()
    second -= second.mean()
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    return float(first @ second / denominator) if denominator else 0.0


def feature_activation_links(inputs, activations, feature_names, targets, top_features=12):
    scores = []
    for index, name in enumerate(feature_names):
        score = max(abs(_pearson(inputs[:, index], activations[:, unit])) for unit in range(activations.shape[1]))
        class_effect = float(np.max(_class_contrast(inputs[:, index:index + 1], targets)))
        scores.append({"index": index, "name": name, "activation_correlation": score, "class_effect": class_effect})
    scores.sort(key=lambda row: row["activation_correlation"], reverse=True)
    groups = {}
    lower_names = [name.lower() for name in feature_names]
    for group, patterns in FEATURE_GROUP_PATTERNS.items():
        group_scores = [row for row, name in zip(scores, lower_names) if any(pattern.lower() in name for pattern in patterns)]
        groups[group] = {
            "feature_count": len(group_scores),
            "mean_top_correlation": float(np.mean([row["activation_correlation"] for row in group_scores[:top_features]])) if group_scores else 0.0,
            "top_features": group_scores[:top_features],
        }
    return {"top_features": scores[:top_features], "groups": groups}


def _zero_hidden_units(model, layer, units):
    result = {name: value.copy() for name, value in model.items()}
    units = list(units)
    if layer == 1:
        result["b1"][units] = 0.0
        result["w1"][:, units] = 0.0
        result["w2"][units, :] = 0.0
    else:
        result["b2"][units] = 0.0
        result["w2"][:, units] = 0.0
        result["w3"][units, :] = 0.0
    return result


def _dynamic_intervention(model, test_x, test_y, units, feature_indices, seed):
    baseline = evaluate_outputs(model, test_x, test_y)
    activation_model = _zero_hidden_units(model, 2, units)
    activation_after = evaluate_outputs(activation_model, test_x, test_y)
    feature_after_inputs = test_x.copy()
    feature_after_inputs[:, feature_indices] = 0.0
    feature_after = evaluate_outputs(model, feature_after_inputs, test_y)
    rng = np.random.default_rng(seed)
    random_units = rng.choice(HIDDEN[1], size=len(units), replace=False)
    random_features = rng.choice(test_x.shape[1], size=len(feature_indices), replace=False)
    random_activation = evaluate_outputs(_zero_hidden_units(model, 2, random_units), test_x, test_y)
    random_feature_inputs = test_x.copy()
    random_feature_inputs[:, random_features] = 0.0
    random_feature = evaluate_outputs(model, random_feature_inputs, test_y)
    return {
        "baseline_accuracy": baseline["accuracy"],
        "selected_activation": {"units": units, "accuracy": activation_after["accuracy"], "loss_change": activation_after["cross_entropy"] - baseline["cross_entropy"]},
        "random_activation": {"units": [int(unit) for unit in random_units], "accuracy": random_activation["accuracy"], "loss_change": random_activation["cross_entropy"] - baseline["cross_entropy"]},
        "selected_features": {"indices": feature_indices, "accuracy": feature_after["accuracy"], "loss_change": feature_after["cross_entropy"] - baseline["cross_entropy"]},
        "random_features": {"indices": [int(index) for index in random_features], "accuracy": random_feature["accuracy"], "loss_change": random_feature["cross_entropy"] - baseline["cross_entropy"]},
    }


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def run_distributed_experiment(data_dir, seeds=(7, 11)):
    data = load_data(data_dir)
    dynamic_train = np.isin(data["train_y"], DYNAMIC_LABELS)
    dynamic_test = np.isin(data["test_y"], DYNAMIC_LABELS)
    models = []
    per_seed = []
    for seed in seeds:
        model = train_baseline(data["train_x"], data["train_y"], seed=seed)
        train_h1, train_h2, _ = baseline_forward(model, data["train_x"])
        test_h1, test_h2, _ = baseline_forward(model, data["test_x"])
        layer_results = []
        for layer, train_values, test_values in ((1, train_h1, test_h1), (2, train_h2, test_h2)):
            train_dynamic_values = train_values[dynamic_train]
            test_dynamic_values = test_values[dynamic_test]
            selected = select_discriminative_units(train_dynamic_values, data["train_y"][dynamic_train])
            concentration = activation_concentration(train_dynamic_values, data["train_y"][dynamic_train])
            profile_train = class_activation_profile(train_dynamic_values, data["train_y"][dynamic_train])
            profile_test = class_activation_profile(test_dynamic_values, data["test_y"][dynamic_test])
            coactive = class_activation_profile((train_dynamic_values > 0).astype(float), data["train_y"][dynamic_train])
            layer_results.append({"layer": layer, "selection": selected, "concentration": concentration, "train_profile": profile_train, "test_profile": profile_test, "coactivation_profile": coactive})
        links = feature_activation_links(data["train_x"][dynamic_train], train_h2[dynamic_train], data["feature_names"], data["train_y"][dynamic_train])
        feature_indices = [row["index"] for row in links["top_features"][:8]]
        intervention = _dynamic_intervention(
            model,
            data["test_x"][dynamic_test], data["test_y"][dynamic_test],
            layer_results[1]["selection"]["units"], feature_indices, seed + 100,
        )
        per_seed.append({"seed": seed, "layers": layer_results, "feature_links": links, "intervention": intervention})
        models.append((model, test_h2[dynamic_test]))
    similarity = {}
    if len(models) >= 2:
        for layer in (1, 2):
            activations = []
            for model, _ in models:
                hidden1, hidden2, _ = baseline_forward(model, data["test_x"][dynamic_test])
                values = hidden1 if layer == 1 else hidden2
                activations.append(values)
            similarity[str(layer)] = {
                "distance_structure_similarity": pattern_similarity(activations[0], activations[1]),
                "coactivation_structure_similarity": pattern_similarity((activations[0] > 0).astype(float), (activations[1] > 0).astype(float)),
                "top_unit_overlap": len(set(per_seed[0]["layers"][layer - 1]["selection"]["units"]) & set(per_seed[1]["layers"][layer - 1]["selection"]["units"])),
            }
    stages = [
        {"name": "seed_activation_patterns", "hypothesis": "같은 활동은 seed가 달라도 비슷한 내부 패턴을 만든다.", "experiment": "동일 test dynamic samples의 layer별 activation distance structure와 coactivation structure 비교", "actual": similarity, "mismatch": "뉴런 ID overlap이 낮아도 distance structure가 높으면 분산 표현의 증거지만, 둘 다 낮으면 공통 표현 가설이 약해진다.", "next_question": "공통 구조가 class별로 유지되는가?"},
        {"name": "common_and_differential_activity", "hypothesis": "세 동적 활동은 공통 운동 표현과 활동별 차별 표현을 함께 가진다.", "experiment": "class centroid distances, active-unit counts, contrast energy 비교", "actual": [{"seed": row["seed"], "layers": [{"layer": layer["layer"], "centroid_distances": layer["test_profile"]["centroid_distances"], "concentration": {key: value for key, value in layer["concentration"].items() if key.endswith("fraction")}} for layer in row["layers"]]} for row in per_seed], "mismatch": "top-1 energy가 크면 분산 가설보다 소수 unit 가설이 강해진다.", "next_question": "차별 표현은 어떤 sensor feature group과 연결되는가?"},
        {"name": "feature_links", "hypothesis": "세부 표현은 entropy/frequency/jerk/autocorrelation feature와 연결된다.", "experiment": "train dynamic subset에서 feature-activation correlation과 class effect 계산", "actual": [{"seed": row["seed"], "groups": {key: value["mean_top_correlation"] for key, value in row["feature_links"]["groups"].items()}, "top_features": [item["name"] for item in row["feature_links"]["top_features"][:8]]} for row in per_seed], "mismatch": "seed별 group ranking이 다르면 특정 feature family의 안정적 의미를 약하게 표현한다.", "next_question": "연결된 feature/activation pattern을 교란하면 세부 분류가 무너지는가?"},
        {"name": "intervention", "hypothesis": "분산 pattern과 연결된 feature/activation을 교란하면 random보다 세부 분류가 더 크게 변한다.", "experiment": "dynamic test subset에서 선택된 units/features와 같은 크기 random control 교란", "actual": [{"seed": row["seed"], "intervention": row["intervention"]} for row in per_seed], "mismatch": "선택 교란 효과가 random보다 작거나 seed마다 방향이 다르면 인과적 해석이 약해진다.", "next_question": "서로 다른 unit 조합이 같은 feature-level 기능을 대체하는가?"},
        {"name": "alternative_circuits", "hypothesis": "seed마다 다른 unit이지만 비슷한 representation geometry와 feature group을 만든다.", "experiment": "top-unit overlap과 permutation-invariant distance/coactivation similarity 비교", "actual": similarity, "mismatch": "geometry similarity가 낮으면 대체 회로 가설을 지지하지 않는다.", "next_question": "더 많은 seed와 별도 모델 구조에서도 반복되는가?"},
        {"name": "reframed_question", "hypothesis": "기능은 뉴런 번호가 아니라 계산 pattern으로 설명해야 한다.", "experiment": "모든 단계 결과를 neuron ID, pattern geometry, feature links, interventions로 분리 기록", "actual": {"interpretation": "뉴런 번호의 재현성보다 pattern-level evidence를 우선한다."}, "mismatch": "현재 두 seed만으로 분산 pattern의 보편성을 증명할 수 없다.", "next_question": "더 많은 seed와 raw temporal model에서 같은 pattern이 재현되는가?"},
    ]
    return _jsonable({"settings": {"seeds": list(seeds), "dynamic_labels": list(DYNAMIC_LABELS)}, "runs": per_seed, "seed_similarity": similarity, "stages": stages})


def write_report(result, path):
    lines = ["# 분산 계산 패턴 분석", "", "가설: 세부 활동 판단은 고정된 뉴런이나 pair가 아니라 여러 뉴런과 시간적 센서 특징의 분산 pattern으로 구현될 수 있다.", ""]
    for stage in result["stages"]:
        lines += [f"## {stage['name']}", "", f"- 가설: {stage['hypothesis']}", f"- 실험: {stage['experiment']}", f"- 실제 결과: `{json.dumps(stage['actual'], ensure_ascii=False)}`", f"- 기존 설명과 맞지 않는 점: {stage['mismatch']}", f"- 수정된 다음 질문: {stage['next_question']}", ""]
    lines += ["## 최종 해석", "", "현재 결과는 특정 뉴런 번호를 기능의 이름으로 삼는 해석을 약화한다. 대신 seed 간 거리 구조·공동 활성화·feature 연결·교란 효과가 함께 재현되는지를 기능의 증거로 사용해야 한다. 이 증거가 약하면 분산 pattern 가설도 확정하지 않는다.", "", "## 재현", "", "`python -m uci_har.distributed_experiment`"]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    root = Path(__file__).parent
    output = run_distributed_experiment(root / "UCI HAR Dataset")
    (root / "distributed_results.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    write_report(output, root / "distributed_analysis.md")
    print(json.dumps({"runs": len(output["runs"]), "result": str(root / "distributed_results.json")}, indent=2))
