"""Explain one MLP representation update with its local Jacobian."""

import json
from pathlib import Path

import numpy as np

from .gradient_geometry_experiment import _copy_model, _cosine, _loss
from .uci_har_experiment import _adam_update, baseline_forward, load_data, new_model


def _forward_parts(model, inputs):
    z1 = inputs @ model["w1"] + model["b1"]
    h1 = np.maximum(z1, 0.0)
    z2 = h1 @ model["w2"] + model["b2"]
    h2 = np.maximum(z2, 0.0)
    probabilities = baseline_forward(model, inputs)[2]
    return z1, h1, z2, h2, probabilities


def jacobian_delta_parts(model, inputs, parameter_delta):
    z1, h1, z2, _, _ = _forward_parts(model, inputs)
    gate1, gate2 = z1 > 0, z2 > 0
    dw1, db1 = parameter_delta["w1"], parameter_delta["b1"]
    dw2, db2 = parameter_delta["w2"], parameter_delta["b2"]
    parts = {
        "w1": ((gate1 * (inputs @ dw1)) @ model["w2"]) * gate2,
        "b1": ((gate1 * db1) @ model["w2"]) * gate2,
        "w2": (h1 @ dw2) * gate2,
        "b2": np.broadcast_to(db2, z2.shape) * gate2,
    }
    return parts, gate1, gate2


def _distance_changes(before, after, targets):
    def distances(values):
        squared = np.sum(values * values, axis=1)[:, None]
        return np.sqrt(np.maximum(squared + squared.T - 2 * values @ values.T, 0.0))
    first, second = distances(before), distances(after)
    upper = np.triu_indices(len(targets), 1)
    same = targets[upper[0]] == targets[upper[1]]
    change = second - first
    same_change, different_change = change[upper][same], change[upper][~same]
    return {"same_class": float(same_change.mean()), "different_class": float(different_change.mean()), "gap": float(different_change.mean() - same_change.mean())}


def _vector_fit(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    error = predicted - actual
    centered = actual - actual.mean(axis=0, keepdims=True)
    return {
        "cosine": float(np.mean([_cosine(actual[i], predicted[i]) for i in range(len(actual))])),
        "r2": float(1.0 - np.sum(error * error) / np.sum(centered * centered)) if np.sum(centered * centered) else 0.0,
        "norm_error": float(np.linalg.norm(error) / np.maximum(np.linalg.norm(actual), 1e-12)),
        "actual_norm": float(np.linalg.norm(actual) / np.sqrt(len(actual))),
        "predicted_norm": float(np.linalg.norm(predicted) / np.sqrt(len(predicted))),
    }


def _gradient_step(model, batch_x, batch_y):
    z1, h1, z2, h2, probabilities = _forward_parts(model, batch_x)
    error = probabilities.copy()
    error[np.arange(len(batch_y)), batch_y] -= 1.0
    error /= len(batch_y)
    dh2 = (error @ model["w3"].T) * (h2 > 0)
    gradients = {"w3": h2.T @ error, "b3": error.sum(axis=0), "w2": h1.T @ dh2, "b2": dh2.sum(axis=0)}
    dh1 = (dh2 @ model["w2"].T) * (h1 > 0)
    gradients.update({"w1": batch_x.T @ dh1, "b1": dh1.sum(axis=0)})
    return h2, probabilities, gradients, z1, z2


def _model_with_delta(model, delta):
    return {name: model[name] + delta[name] for name in model}


def _update_metrics(model_before, model_after, batch_x, batch_y, parameter_delta, label):
    before_h2, before_probabilities, _, before_z1, before_z2 = _gradient_step(model_before, batch_x, batch_y)
    after_h2 = _forward_parts(model_after, batch_x)[3]
    parts, gate1, gate2 = jacobian_delta_parts(model_before, batch_x, parameter_delta)
    predicted_delta = sum(parts.values())
    actual_delta = after_h2 - before_h2
    stable = np.all((_forward_parts(model_after, batch_x)[0] > 0) == gate1, axis=1) & np.all((_forward_parts(model_after, batch_x)[2] > 0) == gate2, axis=1)
    result = {"label": label, "fit": _vector_fit(actual_delta, predicted_delta), "gate_changed_fraction": float(1.0 - np.mean(stable)), "stable_gate_fit": _vector_fit(actual_delta[stable], predicted_delta[stable]) if stable.any() else None, "changed_gate_fit": _vector_fit(actual_delta[~stable], predicted_delta[~stable]) if (~stable).any() else None, "actual_distance_change": _distance_changes(before_h2, after_h2, batch_y), "predicted_distance_change": _distance_changes(before_h2, before_h2 + predicted_delta, batch_y)}
    result["parts"] = {name: {"fit_to_actual": _vector_fit(actual_delta, value), "norm": float(np.linalg.norm(value) / np.sqrt(len(value))), "distance_change": _distance_changes(before_h2, before_h2 + value, batch_y)} for name, value in parts.items()}
    return result


def trace_seed(data, seed, updates=10, batch_size=128, learning_rate=0.001):
    model = new_model(seed)
    rng = np.random.default_rng(seed + 1)
    moments = {name: [np.zeros_like(value), np.zeros_like(value)] for name, value in model.items()}
    records = []
    for update, batch_indices in enumerate(np.array_split(rng.permutation(len(data["train_x"])), max(1, len(data["train_x"]) // batch_size))):
        if update >= updates:
            break
        batch_x, batch_y = data["train_x"][batch_indices], data["train_y"][batch_indices]
        before_h2, before_probabilities, gradients, _, _ = _gradient_step(model, batch_x, batch_y)
        old_model = _copy_model(model)
        _adam_update(model, gradients, moments, update + 1, learning_rate)
        adam_delta = {name: model[name] - old_model[name] for name in model}
        adam_result = _update_metrics(old_model, model, batch_x, batch_y, adam_delta, "adam")
        sgd_delta = {name: -learning_rate * gradients[name] for name in model}
        sgd_model = _model_with_delta(old_model, sgd_delta)
        sgd_result = _update_metrics(old_model, sgd_model, batch_x, batch_y, sgd_delta, "sgd")
        records.append({"update": update + 1, "loss_before": _loss(before_probabilities, batch_y), "loss_after_adam": _loss(_forward_parts(model, batch_x)[4], batch_y), "adam": adam_result, "sgd_counterfactual": sgd_result, "weight_update_cosines": {name: _cosine(-gradients[name], adam_delta[name]) for name in ("w1", "w2", "w3")}})
    return records


def run_jacobian_updates(data_dir, seeds=(7, 11, 19, 23, 31)):
    data = load_data(data_dir)
    runs = [{"seed": seed, "records": trace_seed(data, seed)} for seed in seeds]
    stages = [
        {"prediction": "Δh ≈ Jθ(h)Δθ로 실제 hidden 이동을 설명할 수 있다.", "measurement": "Adam 실제 update의 Jacobian 예측 Δh와 실제 Δh의 cosine, R², norm error를 계산한다."},
        {"prediction": "W1/W2/bias contribution을 합치면 class geometry 변화도 예측된다.", "measurement": "parameter별 1차 contribution과 same/different class distance 변화량을 계산한다."},
        {"prediction": "ReLU gate가 유지된 sample에서는 근사가 더 정확하다.", "measurement": "gate 유지/변경 sample의 Jacobian fit을 분리한다."},
        {"prediction": "Adam의 preconditioning이 단순 SGD와 representation 이동을 다르게 만든다.", "measurement": "동일 gradient에서 Adam 실제 update와 SGD counterfactual의 fit·거리 변화를 비교한다."},
    ]
    adam = [record["adam"] for run in runs for record in run["records"]]
    sgd = [record["sgd_counterfactual"] for run in runs for record in run["records"]]
    stages[0]["actual_result"] = {"adam_cosine": float(np.mean([row["fit"]["cosine"] for row in adam])), "adam_r2": float(np.mean([row["fit"]["r2"] for row in adam])), "adam_norm_error": float(np.mean([row["fit"]["norm_error"] for row in adam])), "adam_predicted_gap": float(np.mean([row["predicted_distance_change"]["gap"] for row in adam])), "adam_actual_gap": float(np.mean([row["actual_distance_change"]["gap"] for row in adam]))}
    stages[1]["actual_result"] = {name: {"norm": float(np.mean([row["parts"][name]["norm"] for row in adam])), "distance_gap": float(np.mean([row["parts"][name]["distance_change"]["gap"] for row in adam]))} for name in ("w1", "b1", "w2", "b2")}
    stages[2]["actual_result"] = {"gate_changed_fraction": float(np.mean([row["gate_changed_fraction"] for row in adam])), "stable_gate_r2": float(np.mean([row["stable_gate_fit"]["r2"] for row in adam if row["stable_gate_fit"] is not None])), "changed_gate_r2": float(np.mean([row["changed_gate_fit"]["r2"] for row in adam if row["changed_gate_fit"] is not None]))}
    stages[3]["actual_result"] = {"sgd_cosine": float(np.mean([row["fit"]["cosine"] for row in sgd])), "sgd_r2": float(np.mean([row["fit"]["r2"] for row in sgd])), "sgd_norm_error": float(np.mean([row["fit"]["norm_error"] for row in sgd])), "adam_w1_gap": stages[1]["actual_result"]["w1"]["distance_gap"], "adam_w2_gap": stages[1]["actual_result"]["w2"]["distance_gap"]}
    return {"settings": {"seeds": list(seeds), "updates_per_seed": 10, "batch_size": 128}, "runs": runs, "stages": stages}


def write_report(result, path):
    lines = ["# Jacobian으로 설명하는 weight update와 hidden 이동", ""]
    for index, stage in enumerate(result["stages"], 1):
        lines += [f"## {index}", "", f"- 수식의 예측: {stage['prediction']}", f"- 실제 측정: {stage['measurement']}", f"- 실제 결과: {stage['actual_result']}", "- 맞지 않는 점: batch 원자료에서 선형근사 오차와 gate 변화 효과를 확인한다.", "- 수정된 메커니즘: `loss → Δθ → Jθ(h)Δθ → 실제 Δh → class 거리 변화`의 각 오차원을 분리한다.", ""]
    lines += ["## 최소 메커니즘", "", "`loss gradient → Adam/SGD parameter update → local Jacobian prediction → ReLU-gated nonlinear residual → class geometry`"]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    root = Path(__file__).parent
    result = run_jacobian_updates(root / "UCI HAR Dataset")
    (root / "jacobian_update_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_report(result, root / "jacobian_update_analysis.md")
    print(json.dumps({"runs": len(result["runs"]), "result": str(root / "jacobian_update_results.json")}, indent=2))
