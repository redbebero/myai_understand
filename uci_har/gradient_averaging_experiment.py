"""Test whether averaging sample gradients creates shared generalizing geometry."""

import json
from pathlib import Path

import numpy as np

from .direct_overlap_inverse_experiment import _direct_metrics
from .generalization_experiment import _adam_update, _copy, _forward, _gradients, _init_model
from .margin_inverse_experiment import _geometry, _margin_metrics
from .validation_selective_inverse_experiment import _strict_split


BATCH_SIZES = (1, 4, 16, 64, 128)
UPDATES = 10
LEARNING_RATE = 0.001


def _names(model):
    return tuple(model)


def _flatten(values, names):
    return np.concatenate([values[name].ravel() for name in names])


def _unflatten(vector, template, names):
    result, offset = {}, 0
    for name in names:
        size = template[name].size
        result[name] = vector[offset:offset + size].reshape(template[name].shape).copy()
        offset += size
    return result


def _loss(model, inputs, targets):
    probabilities = _forward(model, inputs, 2)[2]
    return float(-np.log(np.maximum(probabilities[np.arange(len(targets)), targets], 1e-12)).mean())


def _centroid_signature(model, inputs, targets):
    h2 = _forward(model, inputs, 2)[0][-1]
    centroids = np.asarray([h2[targets == label].mean(axis=0) for label in range(6)])
    centered = centroids - centroids.mean(axis=0)
    distances = np.linalg.norm(centroids[:, None] - centroids[None, :], axis=2)[np.triu_indices(6, 1)]
    return centered.ravel(), distances


def _geometry_similarity(model, validation, test):
    val, val_dist = _centroid_signature(model, validation[0], validation[1])
    test, test_dist = _centroid_signature(model, test[0], test[1])
    return {"centroid_geometry_cosine": float(np.dot(val, test) / max(np.linalg.norm(val) * np.linalg.norm(test), 1e-12)), "pair_distance_gap": float(np.linalg.norm(val_dist - test_dist) / max(np.linalg.norm(val_dist), 1e-12))}


def _sample_gradients(model, inputs, targets):
    names = _names(model)
    rows = []
    for index in range(len(targets)):
        rows.append(_flatten(_gradients(model, inputs[index:index + 1], targets[index:index + 1], 2)[3], names))
    return np.asarray(rows)


def _gradient_decomposition(sample_gradients):
    mean = sample_gradients.mean(axis=0)
    mean_norm = np.linalg.norm(mean)
    unit = mean / max(mean_norm, 1e-12)
    common = (sample_gradients @ unit[:, None]) * unit[None, :]
    residual = sample_gradients - common
    sample_norms = np.linalg.norm(sample_gradients, axis=1)
    residual_norms = np.linalg.norm(residual, axis=1)
    alignment = np.sum(sample_gradients * unit, axis=1) / np.maximum(sample_norms, 1e-12)
    return {"mean": mean, "residual": residual, "alignment": float(np.mean(alignment)), "residual_cancellation": float(1.0 - mean_norm / max(np.mean(sample_norms), 1e-12)), "residual_energy_fraction": float(np.mean(residual_norms) / max(np.mean(sample_norms), 1e-12)), "mean_norm": float(mean_norm)}


def _evaluate(model, data):
    validation = (data["val_x"], data["val_y"])
    test = (data["test_x"], data["test_y"])
    return {"train_loss": _loss(model, data["train_x"], data["train_y"]), "validation_loss": _loss(model, *validation), "test_loss": _loss(model, *test), "validation_geometry": _geometry(model, *validation), "test_geometry": _geometry(model, *test), "validation_overlap": _direct_metrics(model, *validation)["wrong_probability_mass"], "test_overlap": _direct_metrics(model, *test)["wrong_probability_mass"], "validation_accuracy": _margin_metrics(model, *validation)["accuracy"], "test_accuracy": _margin_metrics(model, *test)["accuracy"], "geometry_similarity": _geometry_similarity(model, validation, test)}


def _apply_vector(model, vector, names, scale):
    result = _copy(model)
    delta = _unflatten(vector, model, names)
    for name in names:
        result[name] -= scale * delta[name]
    return result


def _counterfactuals(model, decomposition, adam_delta, data, names, rng):
    target_norm = np.linalg.norm(_flatten(adam_delta, names))
    mean = decomposition["mean"]
    residuals = decomposition["residual"]
    residual = residuals[np.argmax(np.linalg.norm(residuals, axis=1))] if len(residuals) > 1 else np.zeros_like(mean)
    random = rng.normal(size=mean.size)
    result = {}
    for name, direction in (("mean_gradient", mean), ("residual_sample", residual), ("random", random)):
        direction_norm = np.linalg.norm(direction)
        candidate = _copy(model) if direction_norm < 1e-12 else _apply_vector(model, direction, names, target_norm / direction_norm)
        result[name] = {"update_norm": float(target_norm), "validation_accuracy": _margin_metrics(candidate, data["val_x"], data["val_y"])["accuracy"], "test_accuracy": _margin_metrics(candidate, data["test_x"], data["test_y"])["accuracy"], "test_separation": _geometry(candidate, data["test_x"], data["test_y"])["separation_ratio"], "test_overlap": _direct_metrics(candidate, data["test_x"], data["test_y"])["wrong_probability_mass"]}
    return result


def _run_condition(data, seed, batch_size):
    model = _init_model(561, (64, 32), 6, seed)
    baseline = _copy(model)
    names = _names(model)
    moments = {name: [np.zeros_like(value), np.zeros_like(value)] for name, value in model.items()}
    permutation = np.random.default_rng(seed + 1).permutation(len(data["train_y"]))
    records, counterfactual = [], None
    initial = _evaluate(model, data)
    records.append({"update": 0, "metrics": initial, "gradient": {"alignment": 0.0, "residual_cancellation": 0.0, "residual_energy_fraction": 0.0}})
    rng = np.random.default_rng(seed + 1000 + batch_size)
    for update in range(UPDATES):
        indices = permutation[update * batch_size:(update + 1) * batch_size]
        sample_gradients = _sample_gradients(model, data["train_x"][indices], data["train_y"][indices])
        decomposition = _gradient_decomposition(sample_gradients)
        gradients = _unflatten(decomposition["mean"], model, names)
        before = _copy(model)
        _adam_update(model, gradients, moments, update + 1, LEARNING_RATE)
        if update == 0:
            delta = {name: model[name] - before[name] for name in names}
            counterfactual = _counterfactuals(before, decomposition, delta, data, names, rng)
        records.append({"update": update + 1, "metrics": _evaluate(model, data), "gradient": {key: value for key, value in decomposition.items() if key not in ("mean", "residual")}})
    return {"seed": seed, "batch_size": batch_size, "records": records, "counterfactual": counterfactual}


def run_gradient_averaging(data_dir, seeds=(7, 11, 19, 23, 31)):
    data = _strict_split(data_dir)
    results = [{"batch_size": batch_size, "runs": [_run_condition(data, seed, batch_size) for seed in seeds]} for batch_size in BATCH_SIZES]
    summary = {}
    for condition in results:
        size = condition["batch_size"]
        end = [run["records"][-1] for run in condition["runs"]]
        first = [run["records"][1]["gradient"] for run in condition["runs"]]
        counter = [run["counterfactual"] for run in condition["runs"]]
        summary[str(size)] = {"gradient_alignment": float(np.mean([item["alignment"] for item in first])), "residual_cancellation": float(np.mean([item["residual_cancellation"] for item in first])), "residual_energy_fraction": float(np.mean([item["residual_energy_fraction"] for item in first])), "test_accuracy": float(np.mean([item["metrics"]["test_accuracy"] for item in end])), "validation_accuracy": float(np.mean([item["metrics"]["validation_accuracy"] for item in end])), "test_separation": float(np.mean([item["metrics"]["test_geometry"]["separation_ratio"] for item in end])), "test_overlap": float(np.mean([item["metrics"]["test_overlap"] for item in end])), "val_test_geometry_cosine": float(np.mean([item["metrics"]["geometry_similarity"]["centroid_geometry_cosine"] for item in end])), "counterfactual": {name: {metric: float(np.mean([item[name][metric] for item in counter])) for metric in ("validation_accuracy", "test_accuracy", "test_separation", "test_overlap")} for name in ("mean_gradient", "residual_sample", "random")}}
    return {"settings": {"architecture": "561→64→32→6", "seeds": list(seeds), "batch_sizes": list(BATCH_SIZES), "updates": UPDATES, "split": "strict train/validation/test", "optimizer": "Adam cross-entropy", "counterfactual": "same norm as first Adam update"}, "summary": summary, "results": results}


def write_report(result, path):
    lines = ["# Sample-gradient averaging and generalization", "", "동일 초기화에서 sample별 cross-entropy gradient를 계산하고, batch 평균 방향과 평균에서 벗어난 residual을 비교했다. test는 update에 사용하지 않고 평가만 했다.", "", "## Batch-size summary"]
    for size, values in result["summary"].items():
        lines.append(f"- batch {size}: alignment={values['gradient_alignment']:.3f}, cancellation={values['residual_cancellation']:.3f}, test accuracy={values['test_accuracy']:.3f}, test separation={values['test_separation']:.3f}, val/test geometry cosine={values['val_test_geometry_cosine']:.3f}")
    lines += ["", "## 최소 원리", "", "평균 gradient는 sample별 요구의 합의된 방향이지만, 평균만으로 일반화가 보장되지는 않는다. residual은 평균에서 상쇄되며, batch가 커질수록 그 상쇄가 커지는지와 그 결과 geometry/test 성능이 함께 안정화되는지를 원자료로 판단한다.", "", "`sample gradients → common/residual decomposition → averaging → distribution geometry → validation/test sharing → generalization`"]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    root = Path(__file__).parent
    result = run_gradient_averaging(root / "UCI HAR Dataset")
    (root / "gradient_averaging_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_report(result, root / "gradient_averaging_analysis.md")
    print(json.dumps({"conditions": len(result["results"]), "result": str(root / "gradient_averaging_results.json")}, indent=2))
