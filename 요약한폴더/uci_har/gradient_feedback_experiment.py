"""Compare one-shot, frozen-gradient, and recomputed-gradient trajectories."""

import json
from pathlib import Path

import numpy as np

from .generalization_experiment import _adam_update, _copy, _init_model
from .gradient_averaging_experiment import _evaluate, _flatten, _sample_gradients, _unflatten
from .validation_selective_inverse_experiment import _strict_split


SAMPLE_COUNT = 128
LEARNING_RATE = 0.001
CHECKPOINTS = (0, 1, 2, 4, 8, 16, 32, 64, 128)


def _cosine(first, second):
    return float(np.dot(first, second) / max(np.linalg.norm(first) * np.linalg.norm(second), 1e-12))


def _hidden_movement(model, initial, inputs):
    current = _forward_hidden(model, inputs)
    baseline = _forward_hidden(initial, inputs)
    return float(np.linalg.norm(current - baseline) / np.sqrt(len(inputs)))


def _forward_hidden(model, inputs):
    from .generalization_experiment import _forward
    return _forward(model, inputs, 2)[0][-1]


def _record(model, initial, data, update):
    metrics = _evaluate(model, data)
    metrics["validation_hidden_movement"] = _hidden_movement(model, initial, data["val_x"])
    metrics["test_hidden_movement"] = _hidden_movement(model, initial, data["test_x"])
    metrics["update"] = update
    return metrics


def _make_model_delta(model, initial, names):
    return _flatten(model, names) - _flatten(initial, names)


def _apply_same_norm(initial, delta, names, norm):
    candidate = _copy(initial)
    scale = norm / max(np.linalg.norm(delta), 1e-12)
    shifted = _unflatten(scale * delta, initial, names)
    for name in names:
        candidate[name] += shifted[name]
    return candidate


def _run_seed(data, seed):
    initial = _init_model(561, (64, 32), 6, seed)
    names = tuple(initial)
    sequence = np.random.default_rng(seed + 1).permutation(len(data["train_y"]))[:SAMPLE_COUNT]
    inputs, targets = data["train_x"][sequence], data["train_y"][sequence]
    initial_gradients = _sample_gradients(initial, inputs, targets)
    initial_mean = initial_gradients.mean(axis=0)
    frozen = [_unflatten(row, initial, names) for row in initial_gradients]
    one_shot = _copy(initial)
    one_moments = {name: [np.zeros_like(value), np.zeros_like(value)] for name, value in one_shot.items()}
    _adam_update(one_shot, _unflatten(initial_mean, one_shot, names), one_moments, 1, LEARNING_RATE)

    trajectories = {"one_shot": {"model": one_shot, "records": [_record(initial, initial, data, 0), _record(one_shot, initial, data, 1)], "gradient_cosines": [], "feedback_cosines": []}}
    for method, recompute in (("frozen", False), ("recomputed", True)):
        model = _copy(initial)
        moments = {name: [np.zeros_like(value), np.zeros_like(value)] for name, value in model.items()}
        records = [_record(model, initial, data, 0)]
        gradient_cosines, feedback_cosines = [], []
        previous = None
        for update, index in enumerate(range(SAMPLE_COUNT), 1):
            gradients = _sample_gradients(model, inputs[index:index + 1], targets[index:index + 1])[0] if recompute else _flatten(frozen[index], names)
            if previous is not None:
                gradient_cosines.append(_cosine(previous, gradients))
            feedback_cosines.append(_cosine(_flatten(frozen[index], names), gradients))
            previous = gradients
            _adam_update(model, _unflatten(gradients, model, names), moments, update, LEARNING_RATE)
            if update in CHECKPOINTS:
                records.append(_record(model, initial, data, update))
        trajectories[method] = {"model": model, "records": records, "gradient_cosines": gradient_cosines, "feedback_cosines": feedback_cosines}

    final_norms = {method: np.linalg.norm(_make_model_delta(value["model"], initial, names)) for method, value in trajectories.items()}
    common_norm = float(np.mean(list(final_norms.values())))
    initial_direction = -initial_mean
    for method, value in trajectories.items():
        delta = _make_model_delta(value["model"], initial, names)
        value["parameter_movement_norm"] = float(np.linalg.norm(delta))
        value["final_direction_vs_initial_mean"] = _cosine(delta, initial_direction)
        value["same_norm_model"] = _apply_same_norm(initial, delta, names, common_norm)
        value["same_norm_metrics"] = _record(value["same_norm_model"], initial, data, SAMPLE_COUNT)
        del value["model"], value["same_norm_model"]
    return {"seed": seed, "sequence": sequence.tolist(), "initial_gradient_norm": float(np.linalg.norm(initial_mean)), "trajectories": trajectories, "common_final_movement_norm": common_norm}


def run_feedback(data_dir, seeds=(7, 11, 19, 23, 31)):
    data = _strict_split(data_dir)
    runs = [_run_seed(data, seed) for seed in seeds]
    summary = {}
    for method in ("one_shot", "frozen", "recomputed"):
        values = [run["trajectories"][method] for run in runs]
        final = [value["records"][-1] for value in values]
        same = [value["same_norm_metrics"] for value in values]
        cosines = [value["gradient_cosines"] for value in values if value["gradient_cosines"]]
        feedback = [value["feedback_cosines"] for value in values if value["feedback_cosines"]]
        summary[method] = {"parameter_movement_norm": float(np.mean([value["parameter_movement_norm"] for value in values])), "final_direction_vs_initial_mean": float(np.mean([value["final_direction_vs_initial_mean"] for value in values])), "train_loss": float(np.mean([item["train_loss"] for item in final])), "validation_loss": float(np.mean([item["validation_loss"] for item in final])), "test_loss": float(np.mean([item["test_loss"] for item in final])), "validation_accuracy": float(np.mean([item["validation_accuracy"] for item in final])), "test_accuracy": float(np.mean([item["test_accuracy"] for item in final])), "test_separation": float(np.mean([item["test_geometry"]["separation_ratio"] for item in final])), "test_overlap": float(np.mean([item["test_overlap"] for item in final])), "val_test_geometry_cosine": float(np.mean([item["geometry_similarity"]["centroid_geometry_cosine"] for item in final])), "test_hidden_movement": float(np.mean([item["test_hidden_movement"] for item in final])), "gradient_cosine_mean": float(np.mean([np.mean(item) for item in cosines])) if cosines else None, "gradient_cosine_final": float(np.mean([item[-1] for item in cosines])) if cosines else None, "feedback_cosine_mean": float(np.mean([np.mean(item) for item in feedback])) if feedback else None, "feedback_cosine_final": float(np.mean([item[-1] for item in feedback])) if feedback else None, "same_norm_test_accuracy": float(np.mean([item["test_accuracy"] for item in same])), "same_norm_test_separation": float(np.mean([item["test_geometry"]["separation_ratio"] for item in same])), "same_norm_val_test_geometry_cosine": float(np.mean([item["geometry_similarity"]["centroid_geometry_cosine"] for item in same]))}
    return {"settings": {"architecture": "561→64→32→6", "seeds": list(seeds), "sample_count": SAMPLE_COUNT, "optimizer": "Adam cross-entropy", "split": "strict train/validation/test", "sequence": "same first 128 train samples per seed", "checkpoints": list(CHECKPOINTS)}, "summary": summary, "runs": runs}


def write_report(result, path):
    lines = ["# Iterative gradient feedback", "", "동일한 초기화와 동일한 128개 sample을 사용해 one-shot, frozen-gradient sequential, recomputed sequential을 비교했다. test는 update에 사용하지 않았다.", "", "## Final summary"]
    for method, value in result["summary"].items():
        lines.append(f"- {method}: movement={value['parameter_movement_norm']:.3f}, test accuracy={value['test_accuracy']:.3f}, test separation={value['test_separation']:.3f}, val/test cosine={value['val_test_geometry_cosine']:.3f}, final direction cosine={value['final_direction_vs_initial_mean']:.3f}")
    lines += ["", "## 판정", "", "Recomputed가 frozen/one-shot보다 test geometry와 accuracy에서 우수하고, gradient cosine이 update 중 변하면 iterative feedback 가설을 지지한다. 다만 raw 결과는 update 횟수와 movement 차이를 포함하므로 same-norm 결과를 함께 본다.", "", "`current representation → current error → gradient → parameter change → new representation → new gradient → trajectory → unseen generalization`"]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    root = Path(__file__).parent
    result = run_feedback(root / "UCI HAR Dataset")
    (root / "gradient_feedback_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_report(result, root / "gradient_feedback_analysis.md")
    print(json.dumps({"seeds": len(result["runs"]), "result": str(root / "gradient_feedback_results.json")}, indent=2))
