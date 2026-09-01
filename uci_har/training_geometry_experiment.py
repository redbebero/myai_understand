"""Track class-separating geometry while the existing MLP learns."""

import json
from pathlib import Path

import numpy as np

from .geometry_principle_experiment import _accuracy_from_h2, _class_subspace, _remove_subspace, geometry_metrics
from .uci_har_experiment import baseline_forward, load_data, new_model, _adam_update, softmax


CHECKPOINTS = (0, 1, 2, 5, 10, 20, 80)


def _copy_model(model):
    return {name: value.copy() for name, value in model.items()}


def _train_checkpoints(x, y, seed, checkpoints=CHECKPOINTS):
    model = new_model(seed)
    rng = np.random.default_rng(seed + 1)
    moments = {name: [np.zeros_like(value), np.zeros_like(value)] for name, value in model.items()}
    saved = {0: _copy_model(model)}
    step = 0
    max_epoch = max(checkpoints)
    batch_size = 128
    for epoch in range(1, max_epoch + 1):
        for indices in np.array_split(rng.permutation(len(x)), max(1, len(x) // batch_size)):
            batch_x, batch_y = x[indices], y[indices]
            h1, h2, probabilities = baseline_forward(model, batch_x)
            error = probabilities.copy()
            error[np.arange(len(batch_y)), batch_y] -= 1.0
            error /= len(batch_y)
            gradients = {"w3": h2.T @ error, "b3": error.sum(axis=0)}
            dh2 = (error @ model["w3"].T) * (h2 > 0)
            gradients.update({"w2": h1.T @ dh2, "b2": dh2.sum(axis=0)})
            dh1 = (dh2 @ model["w2"].T) * (h1 > 0)
            gradients.update({"w1": batch_x.T @ dh1, "b1": dh1.sum(axis=0)})
            step += 1
            _adam_update(model, gradients, moments, step, 0.001)
        if epoch in checkpoints:
            saved[epoch] = _copy_model(model)
    return saved


def _checkpoint_result(model, data, sample_indices, random_basis, class_rank=5):
    train_h1, train_h2, _ = baseline_forward(model, data["train_x"])
    test_h1, test_h2, _ = baseline_forward(model, data["test_x"])
    metrics = {}
    for name, values in (("input", data["test_x"]), ("hidden1", test_h1), ("hidden2", test_h2)):
        metric = geometry_metrics(values, data["test_y"], sample_indices)
        metric.pop("centroids", None)
        metrics[name] = metric
    class_basis = _class_subspace(train_h2, data["train_y"], class_rank)
    center = train_h2.mean(axis=0)
    class_removed = _remove_subspace(test_h2, class_basis, center)
    random_removed = _remove_subspace(test_h2, random_basis, center)
    return {
        "accuracy": _accuracy_from_h2(model, test_h2, data["test_y"]),
        "geometry": metrics,
        "class_subspace_removed_accuracy": _accuracy_from_h2(model, class_removed, data["test_y"]),
        "random_subspace_removed_accuracy": _accuracy_from_h2(model, random_removed, data["test_y"]),
        "subspace_effect": _accuracy_from_h2(model, test_h2, data["test_y"]) - _accuracy_from_h2(model, class_removed, data["test_y"]),
    }


def _onset(points, key, fraction=0.1):
    initial = points[0][key]
    final = points[-1][key]
    threshold = initial + fraction * (final - initial)
    for point in points:
        if (final >= initial and point[key] >= threshold) or (final < initial and point[key] <= threshold):
            return point["epoch"]
    return points[-1]["epoch"]


def run_training_geometry(data_dir, seeds=(7, 11, 19, 23, 31)):
    data = load_data(data_dir)
    sample_indices = np.random.default_rng(0).choice(len(data["test_y"]), size=600, replace=False)
    runs = []
    for seed in seeds:
        checkpoints = _train_checkpoints(data["train_x"], data["train_y"], seed)
        rng = np.random.default_rng(seed + 1000)
        random_basis, _ = np.linalg.qr(rng.normal(size=(32, 5)))
        points = [{"epoch": epoch, **_checkpoint_result(model, data, sample_indices, random_basis.T)} for epoch, model in sorted(checkpoints.items())]
        for point in points:
            point["hidden2_separation_gain"] = point["geometry"]["hidden2"]["separation_ratio"] - points[0]["geometry"]["hidden2"]["separation_ratio"]
            point["accuracy_gain"] = point["accuracy"] - points[0]["accuracy"]
        runs.append({"seed": seed, "checkpoints": points, "onset_10pct": {"accuracy": _onset(points, "accuracy_gain"), "hidden2_geometry": _onset(points, "hidden2_separation_gain")}})
    stages = [
        {"hypothesis": "성능 향상은 hidden representation의 class-separating geometry 형성과 함께 일어난다.", "experiment": "epoch 0,1,2,5,10,20,80에서 input·hidden1·hidden2의 거리와 분리도를 측정한다."},
        {"hypothesis": "geometry가 accuracy보다 먼저 형성된다.", "experiment": "각 seed에서 초기값 대비 최종 변화의 10%에 도달하는 epoch를 accuracy와 hidden2 separation ratio에 대해 비교한다."},
        {"hypothesis": "초기 geometry도 이미 실제 판단에 사용된다.", "experiment": "각 checkpoint에서 class-separating hidden2 subspace와 random subspace를 제거해 test accuracy를 비교한다."},
        {"hypothesis": "같은 발달 순서가 seed마다 반복된다.", "experiment": "5개 seed의 checkpoint 곡선과 onset epoch를 비교한다."},
    ]
    stages[0]["actual_result"] = "checkpoint별 원자료는 runs에 저장된다."
    stages[1]["actual_result"] = [{"seed": run["seed"], **run["onset_10pct"]} for run in runs]
    stages[2]["actual_result"] = "각 checkpoint의 class/random subspace accuracy가 runs에 저장된다."
    stages[3]["actual_result"] = "seed별 결과와 예외를 곡선 원자료로 비교한다."
    return {"settings": {"seeds": list(seeds), "checkpoints": list(CHECKPOINTS), "onset_fraction": 0.1}, "runs": runs, "stages": stages}


def write_report(result, path):
    lines = ["# 학습 중 class-separating geometry 형성", ""]
    for index, stage in enumerate(result["stages"], 1):
        lines += [f"## {index}", "", f"- 가설: {stage['hypothesis']}", f"- 실험: {stage['experiment']}", f"- 실제 결과: `{json.dumps(stage['actual_result'], ensure_ascii=False)}`", "- 모순: checkpoint 원자료에서 가설과 다른 seed·epoch를 확인한다.", "- 수정된 원리: geometry와 accuracy의 상대적 onset 및 subspace 개입 효과로 판단한다.", ""]
    lines += ["## 최종 판정", "", "각 seed의 checkpoint 원자료를 기준으로 geometry 선행·동시 발달·후행·seed 의존성 중 하나를 선택한다.", ""]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    root = Path(__file__).parent
    result = run_training_geometry(root / "UCI HAR Dataset")
    (root / "training_geometry_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_report(result, root / "training_geometry_analysis.md")
    print(json.dumps({"runs": len(result["runs"]), "result": str(root / "training_geometry_results.json")}, indent=2))
