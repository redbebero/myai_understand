"""Test whether frozen MLP decisions survive hidden-representation compression."""

import json
from pathlib import Path

import numpy as np

from .geometry_principle_experiment import _class_subspace
from .uci_har_experiment import baseline_forward, load_data, softmax, train_baseline
from .validation_selective_inverse_experiment import _strict_split


SEEDS = (7, 11, 19, 23, 31)
DIMENSIONS = (32, 16, 8, 4, 2, 1)
RANDOM_DRAWS = 20
SAMPLE_SIZE = 600


def _project_and_reconstruct(representations, basis, mean):
    centered = representations - mean
    return mean + (centered @ basis.T) @ basis


def _pairwise_distances(values):
    squared = np.sum(values * values, axis=1)[:, None]
    return np.sqrt(np.maximum(squared + squared.T - 2 * values @ values.T, 0.0))


def _pairwise_distance_correlation(first, second):
    upper = np.triu_indices(len(first), 1)
    left = _pairwise_distances(first)[upper]
    right = _pairwise_distances(second)[upper]
    left -= left.mean()
    right -= right.mean()
    return float(np.dot(left, right) / max(np.linalg.norm(left) * np.linalg.norm(right), 1e-12))


def _rank(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def _class_separating_basis(representations, labels, k):
    return _class_subspace(representations, labels, k)


def _pca_basis(representations):
    _, _, vectors = np.linalg.svd(representations - representations.mean(axis=0), full_matrices=False)
    return vectors


def _complete_basis(primary, fallback, k):
    rows = []
    for vector in np.vstack((primary, fallback)):
        candidate = vector.copy()
        for previous in rows:
            candidate -= np.dot(candidate, previous) * previous
        norm = np.linalg.norm(candidate)
        if norm > 1e-10:
            rows.append(candidate / norm)
        if len(rows) == k:
            break
    return np.asarray(rows)


def _output_weights(model):
    index = max(int(name[1:]) for name in model if name.startswith("w"))
    return model[f"w{index}"], model[f"b{index}"]


def _evaluate_method(model, original_hidden, eval_hidden, labels, basis, mean, sample_indices=None):
    compressed = _project_and_reconstruct(eval_hidden, basis, mean)
    weights, bias = _output_weights(model)
    original_probabilities = softmax(original_hidden @ weights + bias)
    compressed_probabilities = softmax(compressed @ weights + bias)
    original_predictions = original_probabilities.argmax(axis=1)
    compressed_predictions = compressed_probabilities.argmax(axis=1)
    if sample_indices is None:
        sample_indices = np.arange(min(len(labels), SAMPLE_SIZE))
    original_sample = original_hidden[sample_indices]
    compressed_sample = compressed[sample_indices]
    centroids = np.array([compressed[labels == label].mean(axis=0) for label in np.unique(labels)])
    global_mean = compressed.mean(axis=0)
    within = np.mean([np.mean((compressed[labels == label] - centroids[i]) ** 2) for i, label in enumerate(np.unique(labels))])
    between = np.mean((centroids - global_mean) ** 2)
    target_probability = np.maximum(compressed_probabilities[np.arange(len(labels)), labels], 1e-12)
    original_target_probability = np.maximum(original_probabilities[np.arange(len(labels)), labels], 1e-12)
    original_distances = _pairwise_distances(original_sample)
    compressed_distances = _pairwise_distances(compressed_sample)
    upper = np.triu_indices(len(sample_indices), 1)
    original_rank = _rank(original_distances[upper])
    compressed_rank = _rank(compressed_distances[upper])
    return {
        "accuracy": float(np.mean(compressed_predictions == labels)),
        "prediction_agreement": float(np.mean(compressed_predictions == original_predictions)),
        "cross_entropy": float(-np.log(target_probability).mean()),
        "original_cross_entropy": float(-np.log(original_target_probability).mean()),
        "reconstruction_mse": float(np.mean((compressed - eval_hidden) ** 2)),
        "separation_ratio": float(between / within) if within else 0.0,
        "distance_correlation": _pairwise_distance_correlation(original_sample, compressed_sample),
        "distance_rank_correlation": float(np.dot(original_rank - original_rank.mean(), compressed_rank - compressed_rank.mean()) / max(np.linalg.norm(original_rank - original_rank.mean()) * np.linalg.norm(compressed_rank - compressed_rank.mean()), 1e-12)),
        "dimension": int(basis.shape[0]),
    }


def _method_basis(method, k, train_hidden, train_labels, model, rng):
    mean = train_hidden.mean(axis=0)
    pca = _pca_basis(train_hidden)
    if method == "random_neuron":
        selected = rng.choice(train_hidden.shape[1], size=k, replace=False)
        basis = np.eye(train_hidden.shape[1])[selected]
    elif method == "random_orthogonal":
        basis = np.linalg.qr(rng.normal(size=(train_hidden.shape[1], k)))[0].T
    elif method == "pca":
        basis = pca[:k]
    elif method == "class_separating":
        basis = _complete_basis(_class_separating_basis(train_hidden, train_labels, min(k, 5)), pca, k)
    elif method == "supervised_output":
        _, _, output_directions = np.linalg.svd(model["w3"].T, full_matrices=False)
        basis = _complete_basis(output_directions, pca, k)
    else:
        raise ValueError(f"unknown method: {method}")
    return basis, mean


def _mean_metrics(metrics):
    keys = metrics[0].keys()
    return {key: float(np.mean([item[key] for item in metrics])) for key in keys}


def run_representation_compression(data_dir, seeds=SEEDS):
    data = _strict_split(data_dir)
    methods = ("random_neuron", "random_orthogonal", "pca", "class_separating", "supervised_output")
    results = {method: {str(k): [] for k in DIMENSIONS} for method in methods}
    baseline_runs = []
    for seed in seeds:
        model = train_baseline(data["train_x"], data["train_y"], seed=seed)
        train_hidden = baseline_forward(model, data["train_x"])[1]
        test_hidden = baseline_forward(model, data["test_x"])[1]
        sample_indices = np.random.default_rng(seed + 5000).choice(len(data["test_y"]), SAMPLE_SIZE, replace=False)
        baseline = _evaluate_method(model, test_hidden, test_hidden, data["test_y"], np.eye(32), train_hidden.mean(axis=0), sample_indices)
        baseline_runs.append({"seed": seed, **baseline})
        for method in methods:
            for k in DIMENSIONS:
                draws = RANDOM_DRAWS if method.startswith("random_") else 1
                metrics = []
                for draw in range(draws):
                    rng = np.random.default_rng(seed * 1000 + draw + 9000)
                    basis, mean = _method_basis(method, k, train_hidden, data["train_y"], model, rng)
                    metrics.append(_evaluate_method(model, test_hidden, test_hidden, data["test_y"], basis, mean, sample_indices))
                results[method][str(k)].append({"seed": seed, "metrics": _mean_metrics(metrics), "draws": draws})
    summary = {"original": _mean_metrics(baseline_runs)}
    for method in methods:
        summary[method] = {k: _mean_metrics([run["metrics"] for run in runs]) for k, runs in results[method].items()}
    return {"settings": {"architecture": "561→64→32→6", "seeds": list(seeds), "dimensions": list(DIMENSIONS), "random_draws_per_seed": RANDOM_DRAWS, "split": "strict train/validation/test", "basis_fit": "train hidden representations only", "output_layer": "frozen"}, "summary": summary, "runs": results, "baseline_runs": baseline_runs}


def write_report(result, path):
    lines = ["# Hidden representation compression", "", "학습된 561→64→32→6 MLP의 마지막 hidden representation(32차원)을 train 표현에서 계산한 basis로 압축했다. 압축 후 32차원으로 복원하고 출력층은 고정했다. test는 basis 학습에 사용하지 않았다.", ""]
    original = result["summary"]["original"]
    lines += [f"원본 test accuracy: {original['accuracy']:.3f}", f"원본 prediction 기준: {original['prediction_agreement']:.3f}", "", "| 방법 | k | accuracy | prediction agreement | distance correlation | separation ratio |", "|---|---:|---:|---:|---:|---:|"]
    for method, values in result["summary"].items():
        if method == "original":
            continue
        for k in DIMENSIONS:
            row = values[str(k)]
            lines.append(f"| {method} | {k} | {row['accuracy']:.3f} | {row['prediction_agreement']:.3f} | {row['distance_correlation']:.3f} | {row['separation_ratio']:.3f} |")
    lines += ["", "## 최소 차원", ""]
    for method, values in result["summary"].items():
        if method == "original":
            continue
        eligible = [k for k in DIMENSIONS if values[str(k)]["accuracy"] >= original["accuracy"] - 0.01]
        lines.append(f"- {method}: {min(eligible) if eligible else '없음'}")
    lines += ["", "## 판정", "", "class-separating 또는 supervised-output 방식이 같은 k에서 random/PCA보다 높은 정확도와 prediction agreement를 유지하면 관계 보존 가설을 지지한다. PCA가 우세하면 전체 분산 구조가 더 중요하다는 뜻이고, 모든 방식이 빠르게 무너지면 고정 출력층과 원래 좌표 정렬이 중요하다는 뜻이다."]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    root = Path(__file__).parent
    result = run_representation_compression(root / "UCI HAR Dataset")
    (root / "representation_compression_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_report(result, root / "representation_compression_analysis.md")
    print(json.dumps(result["summary"], indent=2))
