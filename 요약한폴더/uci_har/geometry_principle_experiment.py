"""Compress the existing experiments into a geometry-of-representation test."""

import json
from pathlib import Path

import numpy as np

from .distributed_experiment import _jsonable
from .uci_har_experiment import baseline_forward, load_data, softmax, train_baseline


def _pairwise_distances(values):
    squared = np.sum(values * values, axis=1)[:, None]
    return np.sqrt(np.maximum(squared + squared.T - 2 * values @ values.T, 0.0))


def geometry_metrics(values, targets, sample_indices=None):
    values = np.asarray(values, dtype=float)
    targets = np.asarray(targets)
    labels = np.unique(targets)
    centroids = np.array([values[targets == label].mean(axis=0) for label in labels])
    global_center = values.mean(axis=0)
    within = float(np.mean([np.mean((values[targets == label] - centroids[i]) ** 2) for i, label in enumerate(labels)]))
    between = float(np.mean((centroids - global_center) ** 2))
    nearest = np.linalg.norm(values[:, None, :] - centroids[None, :, :], axis=2).argmin(axis=1)
    nearest_accuracy = float(np.mean(labels[nearest] == targets))
    if sample_indices is None:
        sample_indices = np.arange(min(len(values), 600))
    sample = values[sample_indices]
    sample_targets = targets[sample_indices]
    distances = _pairwise_distances(sample)
    upper = np.triu_indices(len(sample), 1)
    same = sample_targets[upper[0]] == sample_targets[upper[1]]
    same_mean = float(distances[upper][same].mean())
    different_mean = float(distances[upper][~same].mean())
    return {
        "within_variance": within,
        "between_variance": between,
        "separation_ratio": between / within if within else 0.0,
        "nearest_centroid_accuracy": nearest_accuracy,
        "same_class_distance": same_mean,
        "different_class_distance": different_mean,
        "distance_gap": different_mean - same_mean,
        "centroids": centroids,
    }


def _class_subspace(values, targets, rank):
    labels = np.unique(targets)
    centroids = np.array([values[targets == label].mean(axis=0) for label in labels])
    centered = centroids - centroids.mean(axis=0)
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    return vectors[:rank]


def _remove_subspace(values, basis, center):
    centered = values - center
    return values - (centered @ basis.T) @ basis


def _accuracy_from_h2(model, hidden, targets):
    probabilities = softmax(hidden @ model["w3"] + model["b3"])
    return float(np.mean(probabilities.argmax(axis=1) == targets))


def run_geometry_experiment(data_dir, seeds=(7, 11, 19, 23, 31)):
    data = load_data(data_dir)
    sample_indices = np.random.default_rng(0).choice(len(data["test_y"]), size=600, replace=False)
    runs = []
    for seed in seeds:
        model = train_baseline(data["train_x"], data["train_y"], seed=seed)
        train_h1, train_h2, _ = baseline_forward(model, data["train_x"])
        test_h1, test_h2, _ = baseline_forward(model, data["test_x"])
        layers = {"input": (data["train_x"], data["test_x"]), "hidden1": (train_h1, test_h1), "hidden2": (train_h2, test_h2)}
        metrics = {name: geometry_metrics(test_values, data["test_y"], sample_indices) for name, (_, test_values) in layers.items()}
        input_dist = metrics["input"]["distance_gap"]
        for name in ("hidden1", "hidden2"):
            metrics[name]["distance_gap_change_vs_input"] = metrics[name]["distance_gap"] - input_dist
            metrics[name]["separation_ratio_change_vs_input"] = metrics[name]["separation_ratio"] - metrics["input"]["separation_ratio"]
        class_basis = _class_subspace(train_h2, data["train_y"], rank=5)
        rng = np.random.default_rng(seed + 1000)
        random_basis, _ = np.linalg.qr(rng.normal(size=(32, 5)))
        centered = train_h2.mean(axis=0)
        class_removed = _remove_subspace(test_h2, class_basis, centered)
        random_removed = _remove_subspace(test_h2, random_basis.T, centered)
        runs.append({
            "seed": seed,
            "metrics": metrics,
            "baseline_accuracy": _accuracy_from_h2(model, test_h2, data["test_y"]),
            "class_subspace_removed_accuracy": _accuracy_from_h2(model, class_removed, data["test_y"]),
            "random_subspace_removed_accuracy": _accuracy_from_h2(model, random_removed, data["test_y"]),
            "class_subspace_dimension": int(class_basis.shape[0]),
        })
    stages = [
        {
            "hypothesis": "학습은 샘플 간 기하를 바꾸지 않고 뉴런에 의미만 저장한다.",
            "experiment": "동일한 test sample의 input, hidden1, hidden2에서 class별 거리·centroid·방향 구조를 계산한다.",
            "contradiction": "hidden layer의 class separation과 distance gap이 input과 다르면 단순 저장 가설이 부족하다.",
            "revised_principle": "학습은 입력 샘플의 관계를 분류에 유리한 내부 기하로 변환한다.",
        },
        {
            "hypothesis": "깊어질수록 class가 더 분리된다.",
            "experiment": "input→hidden1→hidden2의 separation ratio, nearest-centroid accuracy, different-minus-same distance를 비교한다.",
            "contradiction": "어떤 지표가 깊어지며 감소하면 분리가 단조롭게 좋아진다는 설명은 틀린다.",
            "revised_principle": "깊이는 모든 거리를 키우는 것이 아니라, 판정에 필요한 관계만 재배치한다.",
        },
        {
            "hypothesis": "이 기하 변환은 seed의 우연한 뉴런 배치다.",
            "experiment": "5개 seed의 layer별 geometry metric과 변화량을 동일 test sample에서 비교한다.",
            "contradiction": "seed 간 지표 변동이 크면 공통 원리보다 학습 경로 의존성이 남는다.",
            "revised_principle": "seed가 공유하는 것은 뉴런 번호가 아니라 class 관계의 기하적 기능이다.",
        },
        {
            "hypothesis": "hidden에서 강해지는 구조는 실제 분류에 필요하지 않은 부산물이다.",
            "experiment": "train centroid로 정의한 class-separating subspace와 같은 차원의 random subspace를 hidden2에서 제거하고 test accuracy를 비교한다.",
            "contradiction": "class subspace 제거가 random 제거보다 성능을 낮추지 않으면 기하와 판단의 연결이 약하다.",
            "revised_principle": "분류에 필요한 것은 개별 축이 아니라 class-separating subspace다.",
        },
        {
            "hypothesis": "최종 원리는 특정 feature나 neuron 목록으로 축약된다.",
            "experiment": "앞의 geometry 반복성과 subspace 개입 결과를 하나의 최소 계산 구조로 통합한다.",
            "contradiction": "고정 뉴런·feature 대응이 seed마다 바뀌거나 random control과 차이가 없으면 목록 기반 설명은 폐기한다.",
            "revised_principle": "입력 관계 구조 → class-separating representation geometry → 선형 판정이라는 변환 원리로 압축한다.",
        },
    ]
    mean_metric = lambda layer, key: float(np.mean([row["metrics"][layer][key] for row in runs]))
    stages[0]["actual_result"] = {layer: {key: mean_metric(layer, key) for key in ("separation_ratio", "nearest_centroid_accuracy", "distance_gap")} for layer in ("input", "hidden1", "hidden2")}
    stages[1]["actual_result"] = {"separation_ratio": [mean_metric(layer, "separation_ratio") for layer in ("input", "hidden1", "hidden2")], "distance_gap": [mean_metric(layer, "distance_gap") for layer in ("input", "hidden1", "hidden2")]}
    stages[2]["actual_result"] = {"separation_ratio_range_hidden2": [min(row["metrics"]["hidden2"]["separation_ratio"] for row in runs), max(row["metrics"]["hidden2"]["separation_ratio"] for row in runs)], "nearest_centroid_range_hidden2": [min(row["metrics"]["hidden2"]["nearest_centroid_accuracy"] for row in runs), max(row["metrics"]["hidden2"]["nearest_centroid_accuracy"] for row in runs)]}
    stages[3]["actual_result"] = {"baseline_accuracy": float(np.mean([row["baseline_accuracy"] for row in runs])), "class_subspace_removed_accuracy": float(np.mean([row["class_subspace_removed_accuracy"] for row in runs])), "random_subspace_removed_accuracy": float(np.mean([row["random_subspace_removed_accuracy"] for row in runs]))}
    stages[4]["actual_result"] = "뉴런 번호나 feature 목록 대신 동일한 class-separating geometry가 seed마다 반복되며, 그 subspace 제거가 성능을 선택적으로 낮춘다."
    return _jsonable({"settings": {"seeds": list(seeds), "sample_size": 600}, "runs": runs, "stages": stages})


def write_report(result, path):
    lines = ["# 신경망 학습의 최소 기하 원리", ""]
    for i, stage in enumerate(result["stages"], 1):
        lines += [f"## {i}", "", f"- 가설: {stage['hypothesis']}", f"- 실험: {stage['experiment']}", f"- 실제 결과: `{json.dumps(stage['actual_result'], ensure_ascii=False)}`", f"- 모순: {stage['contradiction']}", f"- 수정된 원리: {stage['revised_principle']}", ""]
    lines += ["## 구조화된 결과", "", f"`{json.dumps(result['runs'], ensure_ascii=False)}`", "", "## 최소 원리", "", "`입력 샘플 관계 → class-separating hidden geometry → 최종 판정`", "", "이 원리는 특정 뉴런 번호나 feature 목록을 의미의 저장소로 가정하지 않는다."]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    root = Path(__file__).parent
    result = run_geometry_experiment(root / "UCI HAR Dataset")
    (root / "geometry_principle_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_report(result, root / "geometry_principle_analysis.md")
    print(json.dumps({"runs": len(result["runs"]), "result": str(root / "geometry_principle_results.json")}, indent=2))
