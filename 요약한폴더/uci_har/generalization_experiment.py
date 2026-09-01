"""Check whether early-layer geometry dominance generalizes across controlled MLPs."""

import json
from pathlib import Path

import numpy as np

from .jacobian_update_experiment import _distance_changes
from .uci_har_experiment import load_data


SEEDS = (7, 11, 19, 23, 31)
CONFIGS = {
    "adam_baseline": {"optimizer": "adam", "hidden": (64, 32), "input": "raw"},
    "sgd_baseline": {"optimizer": "sgd", "hidden": (64, 32), "input": "raw"},
    "deep": {"optimizer": "adam", "hidden": (64, 64, 32), "input": "raw"},
    "narrow": {"optimizer": "adam", "hidden": (32, 16), "input": "raw"},
    "wide": {"optimizer": "adam", "hidden": (128, 64), "input": "raw"},
    "pca_whiten": {"optimizer": "adam", "hidden": (64, 32), "input": "pca_whiten"},
}


def _softmax(values):
    shifted = values - values.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _init_model(input_size, hidden, classes, seed):
    rng = np.random.default_rng(seed)
    sizes = (input_size,) + tuple(hidden) + (classes,)
    return {f"w{i}": rng.normal(0, np.sqrt(2 / sizes[i]), (sizes[i], sizes[i + 1])) for i in range(len(sizes) - 1)} | {f"b{i}": np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)}


def _forward(model, inputs, hidden_count):
    hs, zs = [inputs], []
    for i in range(hidden_count):
        z = hs[-1] @ model[f"w{i}"] + model[f"b{i}"]
        zs.append(z)
        hs.append(np.maximum(z, 0.0))
    logits = hs[-1] @ model[f"w{hidden_count}"] + model[f"b{hidden_count}"]
    return hs, zs, _softmax(logits)


def _gradients(model, inputs, targets, hidden_count):
    hs, zs, probabilities = _forward(model, inputs, hidden_count)
    error = probabilities.copy()
    error[np.arange(len(targets)), targets] -= 1
    error /= len(targets)
    gradients = {}
    gradients[f"w{hidden_count}"] = hs[-1].T @ error
    gradients[f"b{hidden_count}"] = error.sum(axis=0)
    dh = error @ model[f"w{hidden_count}"].T
    for i in range(hidden_count - 1, -1, -1):
        dz = dh * (zs[i] > 0)
        gradients[f"w{i}"] = hs[i].T @ dz
        gradients[f"b{i}"] = dz.sum(axis=0)
        dh = dz @ model[f"w{i}"].T
    return hs, zs, probabilities, gradients


def _adam_update(model, gradients, moments, step, rate):
    for name, gradient in gradients.items():
        m, v = moments[name]
        m[:] = 0.9 * m + 0.1 * gradient
        v[:] = 0.999 * v + 0.001 * gradient * gradient
        model[name] -= rate * (m / (1 - 0.9**step)) / (np.sqrt(v / (1 - 0.999**step)) + 1e-8)


def _copy(model):
    return {name: value.copy() for name, value in model.items()}


def _parameter_norm(delta, layer):
    return float(np.sqrt(np.sum(delta[f"w{layer}"] ** 2) + np.sum(delta[f"b{layer}"] ** 2)))


def _layer_prediction(model, hs, zs, delta, layer, hidden_count):
    movement = ((hs[layer] @ delta[f"w{layer}"] + delta[f"b{layer}"]) * (zs[layer] > 0))
    for downstream in range(layer + 1, hidden_count):
        movement = (movement @ model[f"w{downstream}"]) * (zs[downstream] > 0)
    return movement


def _apply_layer_delta(model, delta, layer, scale):
    result = _copy(model)
    result[f"w{layer}"] += scale * delta[f"w{layer}"]
    result[f"b{layer}"] += scale * delta[f"b{layer}"]
    return result


def _prepare_inputs(data):
    train, test = data["train_x"], data["test_x"]
    mean = train.mean(axis=0)
    centered = train - mean
    _, singular, vectors = np.linalg.svd(centered, full_matrices=False)
    width = min(128, vectors.shape[0])
    components = vectors[:width]
    transformed_train = centered @ components.T
    scale = transformed_train.std(axis=0)
    scale[scale == 0] = 1
    transformed_test = (test - mean) @ components.T
    return {"raw": (train, test), "pca_whiten": (transformed_train / scale, transformed_test / scale)}


def trace_condition(train_x, test_x, train_y, test_y, config, seed, updates=10, batch_size=128, rate=0.001):
    hidden = config["hidden"]
    hidden_count = len(hidden)
    model = _init_model(train_x.shape[1], hidden, int(train_y.max()) + 1, seed)
    moments = {name: [np.zeros_like(value), np.zeros_like(value)] for name, value in model.items()}
    rng = np.random.default_rng(seed + 1)
    records = []
    for update, indices in enumerate(np.array_split(rng.permutation(len(train_x)), max(1, len(train_x) // batch_size))):
        if update >= updates:
            break
        batch_x, batch_y = train_x[indices], train_y[indices]
        hs, zs, probabilities, gradients = _gradients(model, batch_x, batch_y, hidden_count)
        old = _copy(model)
        if config["optimizer"] == "adam":
            _adam_update(model, gradients, moments, update + 1, rate)
        else:
            for name in model:
                model[name] -= rate * gradients[name]
        delta = {name: model[name] - old[name] for name in model}
        layer_results = {}
        for layer in range(hidden_count):
            predicted = _layer_prediction(old, hs, zs, delta, layer, hidden_count)
            norm = _parameter_norm(delta, layer)
            normalized = _apply_layer_delta(old, delta, layer, 1.0 / max(norm, 1e-12))
            normalized_h2 = _forward(normalized, batch_x, hidden_count)[0][-1]
            predicted_gap = _distance_changes(hs[-1], hs[-1] + predicted, batch_y)["gap"]
            normalized_gap = _distance_changes(hs[-1], normalized_h2, batch_y)["gap"]
            layer_results[f"layer{layer + 1}"] = {
                "parameter_update_norm": norm,
                "predicted_hidden_movement_norm": float(np.linalg.norm(predicted) / np.sqrt(len(predicted))),
                "jacobian_amplification": float(np.linalg.norm(predicted) / np.sqrt(len(predicted)) / max(norm, 1e-12)),
                "predicted_geometry_gain": predicted_gap,
                "geometry_gain_per_parameter_norm": predicted_gap / max(norm, 1e-12),
                "geometry_gain_per_hidden_norm": predicted_gap / max(np.linalg.norm(predicted) / np.sqrt(len(predicted)), 1e-12),
                "same_norm_geometry_gain": normalized_gap,
            }
        records.append({"update": update + 1, "layers": layer_results})
    return records


def run_generalization(data_dir, seeds=SEEDS):
    data = load_data(data_dir)
    prepared = _prepare_inputs(data)
    results = {}
    for name, config in CONFIGS.items():
        train_x, test_x = prepared[config["input"]]
        results[name] = {"config": config, "runs": [{"seed": seed, "records": trace_condition(train_x, test_x, data["train_y"], data["test_y"], config, seed)} for seed in seeds]}
    summary = {}
    for name, result in results.items():
        flat = [record["layers"] for run in result["runs"] for record in run["records"]]
        summary[name] = {layer: {metric: float(np.mean([record[layer][metric] for record in flat])) for metric in ("parameter_update_norm", "jacobian_amplification", "same_norm_geometry_gain", "geometry_gain_per_parameter_norm", "geometry_gain_per_hidden_norm")} for layer in flat[0]}
    return {"settings": {"seeds": list(seeds), "updates_per_condition": 10, "other_dataset": "skipped: sklearn unavailable"}, "summary": summary, "results": results}


def write_report(result, path):
    lines = ["# W1 우세의 적용 범위", "", "다른 표준 dataset은 sklearn이 설치되지 않아 실행하지 않았다.", ""]
    for name, layers in result["summary"].items():
        lines += [f"## {name}", "", f"`{json.dumps(layers, ensure_ascii=False)}`", ""]
    lines += ["## 최소 해석", "", "입력에 가까운 층의 우세가 optimizer, architecture, input geometry에 따라 유지되는지 summary와 원자료를 비교한다."]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    root = Path(__file__).parent
    result = run_generalization(root / "UCI HAR Dataset")
    (root / "generalization_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_report(result, root / "generalization_analysis.md")
    print(json.dumps({"conditions": len(result["results"]), "result": str(root / "generalization_results.json")}, indent=2))
