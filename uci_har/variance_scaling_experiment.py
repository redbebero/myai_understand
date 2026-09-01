"""Explain why Adam's second-moment scaling changes W1 direction."""

import json
from pathlib import Path

import numpy as np

from .adam_decomposition_experiment import _operation_metrics
from .generalization_experiment import _adam_update, _copy, _gradients, _init_model
from .input_geometry_experiment import _alignment, _class_basis, input_conditions


OPERATIONS = ("momentum", "full_adam", "scalar_v", "clipped_v")


def _row_norms(matrix):
    return np.linalg.norm(matrix, axis=1)


def _pearson(first, second):
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    first, second = first - first.mean(), second - second.mean()
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    return float(first @ second / denominator) if denominator else 0.0


def _candidate_directions(g, m_hat, v_hat):
    scalar = np.sqrt(np.mean(v_hat))
    low, high = np.quantile(v_hat, (0.1, 0.9))
    clipped = np.clip(v_hat, low, high)
    return {
        "momentum": -m_hat,
        "full_adam": -m_hat / (np.sqrt(v_hat) + 1e-8),
        "scalar_v": -m_hat / max(scalar, 1e-8),
        "clipped_v": -m_hat / (np.sqrt(clipped) + 1e-8),
    }, {"scalar_v": float(scalar), "clip_low": float(low), "clip_high": float(high)}


def _coordinate_stats(g, m_hat, v_hat, basis):
    contribution = _row_norms(basis.T @ (basis @ g))
    gradient_magnitude = _row_norms(g)
    v_row = v_hat.mean(axis=1)
    scaling = _row_norms(g / (np.sqrt(v_hat) + 1e-8)) / np.maximum(gradient_magnitude, 1e-12)
    effective = _row_norms(m_hat / (np.sqrt(v_hat) + 1e-8)) / np.maximum(_row_norms(m_hat), 1e-12)
    active = gradient_magnitude > 1e-8
    top = contribution >= np.quantile(contribution, 0.9)
    rest = ~top
    def group_mean(values, mask):
        return float(np.mean(values[mask])) if mask.any() else 0.0

    def group_median(values, mask):
        return float(np.median(values[mask])) if mask.any() else 0.0

    return {
        "v_quantiles": [float(value) for value in np.quantile(v_hat, (0.0, 0.25, 0.5, 0.75, 1.0))],
        "v_coefficient_variation": float(v_hat.std() / max(v_hat.mean(), 1e-12)),
        "active_coordinate_fraction": float(active.mean()),
        "class_contribution_v_correlation": _pearson(contribution, v_row),
        "class_contribution_scaling_correlation": _pearson(contribution[active], scaling[active]) if active.sum() > 1 else 0.0,
        "top_class_coordinates": int(top.sum()),
        "top_vs_rest": {
            "class_contribution": [group_mean(contribution, top), group_mean(contribution, rest)],
            "gradient_magnitude": [group_mean(gradient_magnitude, top), group_mean(gradient_magnitude, rest)],
            "v": [group_mean(v_row, top), group_mean(v_row, rest)],
            "scaling_factor": [group_median(scaling, top & active), group_median(scaling, rest & active)],
            "effective_momentum_scale": [group_median(effective, top & active), group_median(effective, rest & active)],
        },
    }


def trace_seed(train_x, train_y, seed, updates=10, batch_size=128):
    model = _init_model(train_x.shape[1], (64, 32), 6, seed)
    rng = np.random.default_rng(seed + 1)
    moments = {name: [np.zeros_like(value), np.zeros_like(value)] for name, value in model.items()}
    basis = _class_basis(train_x, train_y)
    records = []
    for update, indices in enumerate(np.array_split(rng.permutation(len(train_x)), max(1, len(train_x) // batch_size))):
        if update >= updates:
            break
        batch_x, batch_y = train_x[indices], train_y[indices]
        hs, _, _, gradients = _gradients(model, batch_x, batch_y, 2)
        g = gradients["w0"]
        m_prev, v_prev = moments["w0"]
        m = 0.9 * m_prev + 0.1 * g
        v = 0.999 * v_prev + 0.001 * g * g
        m_hat = m / (1 - 0.9 ** (update + 1))
        v_hat = v / (1 - 0.999 ** (update + 1))
        candidates, controls = _candidate_directions(g, m_hat, v_hat)
        operations = {}
        for name, candidate in candidates.items():
            operations[name] = _operation_metrics(model, batch_x, batch_y, hs[-1], basis, candidate)
        records.append({"update": update + 1, "v_stats": _coordinate_stats(g, m_hat, v_hat, basis), "controls": controls, "operations": operations})
        _adam_update(model, gradients, moments, update + 1, 0.001)
    return records


def run_variance_scaling(data_dir, seeds=(7, 11, 19, 23, 31)):
    conditions = input_conditions(data_dir)
    results = {}
    for condition in ("scale_only", "decorrelated"):
        train_x, _, train_y, _ = conditions[condition]
        results[condition] = {"runs": [{"seed": seed, "records": trace_seed(train_x, train_y, seed)} for seed in seeds]}
    summary = {}
    for condition, result in results.items():
        rows = [row for run in result["runs"] for row in run["records"]]
        summary[condition] = {
            "v": {"coefficient_variation": float(np.mean([row["v_stats"]["v_coefficient_variation"] for row in rows])), "class_contribution_v_correlation": float(np.mean([row["v_stats"]["class_contribution_v_correlation"] for row in rows])), "class_contribution_scaling_correlation": float(np.mean([row["v_stats"]["class_contribution_scaling_correlation"] for row in rows]))},
            "top_vs_rest": {key: [float(np.mean([row["v_stats"]["top_vs_rest"][key][i] for row in rows])) for i in (0, 1)] for key in ("class_contribution", "gradient_magnitude", "v", "scaling_factor", "effective_momentum_scale")},
            "operations": {operation: {metric: float(np.mean([row["operations"][operation][metric] for row in rows])) for metric in ("alignment", "same_norm_geometry_gain", "hidden_class_distance_gap_change")} for operation in OPERATIONS},
        }
    stages = [
        {"hypothesis": "decorrelated 입력은 class-important gradient coordinate에 더 큰 v_t를 만든다.", "control": "scale-only와 decorrelated의 v_t 분포 및 top class-coordinate를 비교한다."},
        {"hypothesis": "큰 v_t가 class-important coordinate를 선택적으로 축소한다.", "control": "top class-contribution coordinate와 나머지의 v_t·scaling factor를 비교한다."},
        {"hypothesis": "coordinate-wise scaling이 class direction을 보존하지 못한다.", "control": "momentum, full Adam, scalar-v, clipped-v의 alignment와 same-norm gain을 비교한다."},
    ]
    stages[0]["actual_result"] = {condition: summary[condition]["v"] for condition in summary}
    stages[1]["actual_result"] = {condition: summary[condition]["top_vs_rest"] for condition in summary}
    stages[2]["actual_result"] = {condition: summary[condition]["operations"] for condition in summary}
    return {"settings": {"seeds": list(seeds), "updates_per_condition": 10, "operations": list(OPERATIONS)}, "summary": summary, "results": results, "stages": stages}


def write_report(result, path):
    lines = ["# Adam second-moment scaling과 class-direction 손실", ""]
    for index, stage in enumerate(result["stages"], 1):
        lines += [f"## {index}", "", f"- 가설: {stage['hypothesis']}", f"- 통제 실험: {stage['control']}", f"- 실제 결과: `{json.dumps(stage['actual_result'], ensure_ascii=False)}`", "- 맞지 않는 점: v 크기·coordinate 불균형·class-coordinate 억제를 분리해 확인한다.", "- 수정된 원리: second-moment scaling이 class-aligned gradient를 coordinate-wise로 재가중하는 단계로 설명한다.", ""]
    lines += ["## 최소 메커니즘", "", "`input geometry → gradient coordinate structure → v_t → 1/√v_t scaling → class-direction loss → hidden geometry 감소`"]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    root = Path(__file__).parent
    result = run_variance_scaling(root / "UCI HAR Dataset")
    (root / "variance_scaling_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_report(result, root / "variance_scaling_analysis.md")
    print(json.dumps({"conditions": len(result["results"]), "result": str(root / "variance_scaling_results.json")}, indent=2))
