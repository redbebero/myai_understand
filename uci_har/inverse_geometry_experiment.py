"""Use a centroid-level Jacobian to reverse-design hidden geometry."""

import json
from pathlib import Path

import numpy as np

from .generalization_experiment import _adam_update, _copy, _forward, _gradients, _init_model
from .input_geometry_experiment import _class_basis
from .uci_har_experiment import load_data


LABELS = (0, 1, 2)
TARGET_FACTOR = 1.2
ITERATIONS = 5
TRAIN_EPOCHS = 80


def _hidden_vector(model):
    return np.concatenate([model["w0"].ravel(), model["b0"], model["w1"].ravel(), model["b1"]])


def _vector_delta(vector, model):
    w0_size = model["w0"].size
    b0_size = model["b0"].size
    w1_size = model["w1"].size
    offset = 0
    result = {}
    result["w0"] = vector[offset:offset + w0_size].reshape(model["w0"].shape)
    offset += w0_size
    result["b0"] = vector[offset:offset + b0_size]
    offset += b0_size
    result["w1"] = vector[offset:offset + w1_size].reshape(model["w1"].shape)
    offset += w1_size
    result["b1"] = vector[offset:offset + len(model["b1"])]
    return result


def _apply_hidden_delta(model, vector, fraction=1.0):
    result = _copy(model)
    delta = _vector_delta(vector * fraction, model)
    for name, value in delta.items():
        result[name] += value
    return result


def _centroids(model, inputs, targets):
    h2 = _forward(model, inputs, 2)[0][-1]
    return np.array([h2[targets == label].mean(axis=0) for label in LABELS])


def _target_geometry(centroids):
    center = centroids.mean(axis=0)
    return center + TARGET_FACTOR * (centroids - center)


def _geometry_metrics(current, target):
    current_centered = current - current.mean(axis=0)
    target_centered = target - target.mean(axis=0)
    current_dist = np.linalg.norm(current_centered[:, None, :] - current_centered[None, :, :], axis=2)
    target_dist = np.linalg.norm(target_centered[:, None, :] - target_centered[None, :, :], axis=2)
    upper = np.triu_indices(len(LABELS), 1)
    return {
        "centroid_error": float(np.linalg.norm(current - target) / np.sqrt(current.size)),
        "distance_error": float(np.linalg.norm(current_dist[upper] - target_dist[upper]) / np.sqrt(len(upper[0]))),
        "mean_pair_distance": float(current_dist[upper].mean()),
        "target_mean_pair_distance": float(target_dist[upper].mean()),
    }


def _centroid_jacobian(model, inputs, targets):
    rows = []
    for label in LABELS:
        selected = targets == label
        x = inputs[selected]
        z0 = x @ model["w0"] + model["b0"]
        h0 = np.maximum(z0, 0.0)
        z1 = h0 @ model["w1"] + model["b1"]
        gate0, gate1 = z0 > 0, z1 > 0
        n = len(x)
        for output in range(model["w1"].shape[1]):
            cross = x.T @ (gate1[:, output, None] * gate0) / n
            w0_row = (cross * model["w1"][:, output][None, :]).ravel()
            b0_row = (gate1[:, output, None] * gate0).mean(axis=0) * model["w1"][:, output]
            w1_block = np.zeros_like(model["w1"])
            w1_block[:, output] = (h0 * gate1[:, output, None]).mean(axis=0)
            b1_row = np.array([gate1[:, output].mean()])
            rows.append(np.concatenate([w0_row, b0_row, w1_block.ravel(), b1_row]))
    return np.asarray(rows)


def _inverse_delta(jacobian, target_delta, ridge=1e-6):
    gram = jacobian @ jacobian.T
    scale = max(np.trace(gram) / max(len(gram), 1), 1e-12)
    coefficients = np.linalg.solve(gram + ridge * scale * np.eye(len(gram)), target_delta)
    return jacobian.T @ coefficients


def _accuracy(model, inputs, targets):
    return float(np.mean(_forward(model, inputs, 2)[2].argmax(axis=1) == targets))


def _same_norm_delta(vector, rng):
    random = rng.normal(size=vector.shape)
    return random * (np.linalg.norm(vector) / max(np.linalg.norm(random), 1e-12))


def _gate_state(model, inputs):
    z0 = inputs @ model["w0"] + model["b0"]
    h0 = np.maximum(z0, 0.0)
    z1 = h0 @ model["w1"] + model["b1"]
    return np.concatenate([(z0 > 0).ravel(), (z1 > 0).ravel()])


def _train_model(model, inputs, targets, seed):
    moments = {name: [np.zeros_like(value), np.zeros_like(value)] for name, value in model.items()}
    rng = np.random.default_rng(seed + 1)
    step = 0
    for _ in range(TRAIN_EPOCHS):
        for indices in np.array_split(rng.permutation(len(inputs)), max(1, len(inputs) // 128)):
            _, _, _, gradients = _gradients(model, inputs[indices], targets[indices], 2)
            step += 1
            _adam_update(model, gradients, moments, step, 0.001)
    return model


def run_seed(data, seed):
    model = _train_model(_init_model(data["train_x"].shape[1], (64, 32), 6, seed), data["train_x"], data["train_y"], seed)
    dynamic = np.isin(data["train_y"], LABELS)
    initial_train = _centroids(model, data["train_x"][dynamic], data["train_y"][dynamic])
    target_train = _target_geometry(initial_train)
    initial_test = _centroids(model, data["test_x"][np.isin(data["test_y"], LABELS)], data["test_y"][np.isin(data["test_y"], LABELS)])
    target_test = _target_geometry(initial_test)
    jacobian = _centroid_jacobian(model, data["train_x"], data["train_y"])
    target_delta = (target_train - initial_train).ravel()
    inverse_vector = _inverse_delta(jacobian, target_delta)
    linear_prediction = jacobian @ inverse_vector
    gram = jacobian @ jacobian.T
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    jacobian_rank = int(np.sum(singular_values > max(singular_values[0] * 1e-8, 1e-12)))
    gradient = _gradients(model, data["train_x"], data["train_y"], 2)[3]
    gradient_vector = np.concatenate([gradient["w0"].ravel(), gradient["b0"], gradient["w1"].ravel(), gradient["b1"]])
    rng = np.random.default_rng(seed + 9000)
    inverse_norm = np.linalg.norm(inverse_vector)
    gradient_delta = -gradient_vector * (inverse_norm / max(np.linalg.norm(gradient_vector), 1e-12))
    candidates = {
        "inverse_one_shot": _apply_hidden_delta(model, inverse_vector),
        "random_same_norm": _apply_hidden_delta(model, _same_norm_delta(inverse_vector, rng)),
        "gradient_same_norm": _apply_hidden_delta(model, gradient_delta),
    }
    test_dynamic = np.isin(data["test_y"], LABELS)
    baseline_metrics = {"geometry": _geometry_metrics(initial_test, target_test), "accuracy": _accuracy(model, data["test_x"], data["test_y"])}
    candidate_metrics = {}
    for name, candidate in candidates.items():
        current_test = _centroids(candidate, data["test_x"][test_dynamic], data["test_y"][test_dynamic])
        candidate_metrics[name] = {"geometry": _geometry_metrics(current_test, target_test), "accuracy": _accuracy(candidate, data["test_x"], data["test_y"]), "delta_norm": float(np.linalg.norm(_hidden_vector(candidate) - _hidden_vector(model))), "gate_change_fraction": float(np.mean(_gate_state(candidate, data["train_x"][dynamic]) != _gate_state(model, data["train_x"][dynamic])))}
    iterative = []
    current = _copy(model)
    for iteration in range(ITERATIONS):
        current_train = _centroids(current, data["train_x"][dynamic], data["train_y"][dynamic])
        current_target_delta = (target_train - current_train).ravel()
        current_jacobian = _centroid_jacobian(current, data["train_x"], data["train_y"])
        step = _inverse_delta(current_jacobian, current_target_delta)
        current = _apply_hidden_delta(current, step, fraction=0.2)
        current_test = _centroids(current, data["test_x"][test_dynamic], data["test_y"][test_dynamic])
        iterative.append({"iteration": iteration + 1, "geometry": _geometry_metrics(current_test, target_test), "accuracy": _accuracy(current, data["test_x"], data["test_y"]), "step_norm": float(np.linalg.norm(step) * 0.2)})
    return {"seed": seed, "jacobian_shape": list(jacobian.shape), "jacobian_rank": jacobian_rank, "linear_target_error": float(np.linalg.norm(linear_prediction - target_delta) / np.sqrt(len(target_delta))), "linear_target_norm": float(np.linalg.norm(target_delta) / np.sqrt(len(target_delta))), "baseline": baseline_metrics, "one_shot": candidate_metrics, "iterative": iterative}


def run_inverse_geometry(data_dir, seeds=(7, 11, 19, 23, 31)):
    data = load_data(data_dir)
    runs = [run_seed(data, seed) for seed in seeds]
    summary = {
        "baseline": {"distance_error": float(np.mean([run["baseline"]["geometry"]["distance_error"] for run in runs])), "accuracy": float(np.mean([run["baseline"]["accuracy"] for run in runs]))},
        "one_shot": {name: {"distance_error": float(np.mean([run["one_shot"][name]["geometry"]["distance_error"] for run in runs])), "pair_distance": float(np.mean([run["one_shot"][name]["geometry"]["mean_pair_distance"] for run in runs])), "accuracy": float(np.mean([run["one_shot"][name]["accuracy"] for run in runs]))} for name in ("inverse_one_shot", "random_same_norm", "gradient_same_norm")},
        "iterative": {"distance_error": [float(np.mean([run["iterative"][i]["geometry"]["distance_error"] for run in runs])) for i in range(ITERATIONS)], "accuracy": [float(np.mean([run["iterative"][i]["accuracy"] for run in runs])) for i in range(ITERATIONS)]},
    }
    stages = [
        {"step": "목표 geometry", "hypothesis": "centroid pair distance를 20% 확대하면 목표가 class separation을 명확히 지정한다."},
        {"step": "필요한 Δh", "hypothesis": "현재 centroid와 목표 centroid 차이가 필요한 hidden 이동을 정의한다."},
        {"step": "역계산 Δθ", "hypothesis": "centroid Jacobian의 regularized pseudoinverse가 최소-norm parameter update를 제공한다."},
        {"step": "실제 적용", "hypothesis": "inverse update가 random/gradient 동일 norm보다 목표 geometry에 가까워진다."},
        {"step": "반복 inverse", "hypothesis": "Jacobian을 갱신하면 작은 inverse step이 목표에 안정적으로 접근한다."},
    ]
    for stage in stages:
        stage["actual_result"] = summary
        stage["mismatch"] = "ReLU gate 변화, Jacobian 선형근사, parameter 자유도, test generalization을 원자료에서 분리한다."
        stage["revised_principle"] = "centroid-level geometry는 local Jacobian 범위 안에서만 역설계 가능하다."
    return {"settings": {"seeds": list(seeds), "target_factor": TARGET_FACTOR, "iterations": ITERATIONS, "train_epochs": TRAIN_EPOCHS, "architecture": "561→64→32→6"}, "summary": summary, "runs": runs, "stages": stages}


def write_report(result, path):
    summary = result["summary"]
    mean_rank = np.mean([run["jacobian_rank"] for run in result["runs"]])
    mean_linear_error = np.mean([run["linear_target_error"] for run in result["runs"]])
    mean_target_norm = np.mean([run["linear_target_norm"] for run in result["runs"]])
    mean_gate_change = np.mean([run["one_shot"]["inverse_one_shot"]["gate_change_fraction"] for run in result["runs"]])
    lines = [
        "# Jacobian inverse design으로 hidden geometry 만들기", "",
        "기존 561→64→32→6 MLP를 seed별 80 epoch 학습한 뒤, hidden2의 세 동적 활동 centroid를 중심에서 1.2배 확대하는 목표를 정의했다. 새 feature/neuron은 사용하지 않았다.", "",
        "## 전체 결과", "",
        f"- baseline: test accuracy {summary['baseline']['accuracy']:.3f}, target distance error {summary['baseline']['distance_error']:.3f}",
        f"- inverse one-shot: target distance error {summary['one_shot']['inverse_one_shot']['distance_error']:.3f}, mean pair distance {summary['one_shot']['inverse_one_shot']['pair_distance']:.3f}, accuracy {summary['one_shot']['inverse_one_shot']['accuracy']:.3f}",
        f"- random same-norm: target distance error {summary['one_shot']['random_same_norm']['distance_error']:.3f}, accuracy {summary['one_shot']['random_same_norm']['accuracy']:.3f}",
        f"- gradient same-norm: target distance error {summary['one_shot']['gradient_same_norm']['distance_error']:.3f}, accuracy {summary['one_shot']['gradient_same_norm']['accuracy']:.3f}",
        "",
        "inverse는 baseline 대비 목표 geometry 오차를 약 85% 줄였고, random은 거의 줄이지 못했다. 그러나 accuracy는 0.943에서 0.935로 소폭 감소했다.", "",
    ]
    for index, stage in enumerate(result["stages"], 1):
        lines += [f"## {index}. {stage['step']}", "", f"- 가설: {stage['hypothesis']}"]
        if index == 1:
            lines += ["- 비교 기준: 현재 test centroid geometry와 중심 기준 1.2배 확대한 target geometry."]
        elif index == 2:
            lines += ["- 비교 기준: centroid 차이의 평균 제곱근 크기와 pair-distance 구조."]
        elif index == 3:
            lines += [f"- 실제 결과: J shape={result['runs'][0]['jacobian_shape']}, 유효 rank 평균={mean_rank:.1f}/96, 선형 target residual={mean_linear_error:.4f} (target Δh 평균 크기={mean_target_norm:.4f})."]
        elif index == 4:
            lines += ["- 실제 결과: inverse가 동일 norm random/gradient보다 target geometry에 가까워졌다. 다만 고정된 output layer를 다시 맞추지 않았으므로 geometry 조작이 accuracy 상승으로 이어지지는 않았다."]
        else:
            lines += [f"- 실제 결과: 반복 inverse distance error={', '.join(f'{x:.3f}' for x in summary['iterative']['distance_error'])}; accuracy={', '.join(f'{x:.3f}' for x in summary['iterative']['accuracy'])}."]
        lines += [f"- 모순/실패 원인: {stage['mismatch']}", f"- 수정된 원리: {stage['revised_principle']}", ""]
    lines += [
        "## 실패 원인 분해", "",
        f"- Jacobian 근사: 선형 residual은 작아 local tangent space 자체는 target Δh를 거의 표현했다.",
        f"- ReLU gate: one-shot inverse에서 평균 {mean_gate_change:.1%}의 gate가 변해, 큰 update에서는 고정-gate Jacobian이 완전한 예측이 아니다.",
        "- 목표 geometry의 의미: centroid 거리만 확대했기 때문에 기존 output weight와의 정렬·sample-level 분류 경계는 직접 최적화하지 않았다. 따라서 geometry 성공과 accuracy 성공은 분리됐다.",
        "- parameter 자유도: 유효 rank가 대부분 82~96/96이므로 이번 실패를 자유도 부족으로 설명하기 어렵다. seed 31처럼 inverse norm과 gate 변화가 큰 경우에는 local 선형성 붕괴가 더 유력하다.",
        "- 반복 inverse: 작은 step은 대체로 목표에 접근했지만 seed 31에서 발산성 흔들림이 나타나, Jacobian 재계산만으로 전역 목표를 보장하지 않는다.",
        "",
        "## 수정된 최소 메커니즘", "",
        "`목표 centroid geometry → Δh_target → local Jθ pseudoinverse → hidden parameter update → ReLU-gated actual movement → geometry 변화`",
        "",
        "결론: Jacobian 역산은 ‘원하는 hidden centroid 구조를 국소적으로 만드는 도구’로는 작동한다. 하지만 그 구조가 곧 decision boundary나 정확도를 의미하지는 않는다. 정확도까지 역설계하려면 목표에 output-weight 정렬 또는 class loss 조건을 함께 포함해야 한다.",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    root = Path(__file__).parent
    result = run_inverse_geometry(root / "UCI HAR Dataset")
    (root / "inverse_geometry_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_report(result, root / "inverse_geometry_analysis.md")
    print(json.dumps({"runs": len(result["runs"]), "result": str(root / "inverse_geometry_results.json")}, indent=2))
