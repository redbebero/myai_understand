"""Compare direct error minimization with an error-increase detour."""

import json
from pathlib import Path

import numpy as np

from .generalization_experiment import _adam_update, _copy, _forward, _gradients, _init_model
from .validation_selective_inverse_experiment import _strict_split


SEEDS = (7, 11, 19, 23, 31)
UPDATES = 20
ASCENT_UPDATES = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.001
HIDDEN = (64, 32)


def _loss(model, inputs, targets, hidden_count=2):
    probabilities = _forward(model, inputs, hidden_count)[2]
    return float(-np.log(np.maximum(probabilities[np.arange(len(targets)), targets], 1e-12)).mean())


def _accuracy(model, inputs, targets):
    return float(np.mean(_forward(model, inputs, len(HIDDEN))[2].argmax(axis=1) == targets))


def _record(model, data, phase, update):
    return {
        "update": update,
        "phase": phase,
        "train_loss": _loss(model, data["train_x"], data["train_y"]),
        "validation_loss": _loss(model, data["val_x"], data["val_y"]),
        "test_loss": _loss(model, data["test_x"], data["test_y"]),
        "train_accuracy": _accuracy(model, data["train_x"], data["train_y"]),
        "validation_accuracy": _accuracy(model, data["val_x"], data["val_y"]),
        "test_accuracy": _accuracy(model, data["test_x"], data["test_y"]),
    }


def _run_condition(data, seed, updates=UPDATES, ascent_updates=0, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE):
    model = _init_model(data["train_x"].shape[1], HIDDEN, 6 if data["train_y"].max() > 1 else 2, seed)
    baseline = _copy(model)
    moments = {name: [np.zeros_like(value), np.zeros_like(value)] for name, value in model.items()}
    rng = np.random.default_rng(seed + 1)
    batches = []
    while len(batches) < updates:
        batches.extend(np.array_split(rng.permutation(len(data["train_y"])), max(1, int(np.ceil(len(data["train_y"]) / batch_size)))))
    batches = batches[:updates]
    records = [_record(model, data, "start", 0)]
    for update, indices in enumerate(batches, 1):
        _, _, _, gradients = _gradients(model, data["train_x"][indices], data["train_y"][indices], len(HIDDEN))
        phase = "ascent" if update <= ascent_updates else "descent"
        if phase == "ascent":
            gradients = {name: -value for name, value in gradients.items()}
        _adam_update(model, gradients, moments, update, learning_rate)
        records.append(_record(model, data, phase, update))
    final_delta = {name: model[name] - baseline[name] for name in model}
    return {"seed": seed, "records": records, "final_parameter_movement": float(np.sqrt(sum(np.sum(value * value) for value in final_delta.values())))}


def _summary(runs):
    return [
        {
            "update": update,
            "phase": runs[0]["records"][update]["phase"],
            **{metric: float(np.mean([run["records"][update][metric] for run in runs])) for metric in ("train_loss", "validation_loss", "test_loss", "train_accuracy", "validation_accuracy", "test_accuracy")},
        }
        for update in range(len(runs[0]["records"]))
    ]


def run_error_detour(data_dir, seeds=SEEDS):
    data = _strict_split(data_dir)
    conditions = {
        "minimize_only": {"updates": UPDATES, "ascent_updates": 0},
        "ascent_then_descent": {"updates": UPDATES, "ascent_updates": ASCENT_UPDATES},
    }
    results = {}
    for name, settings in conditions.items():
        runs = [_run_condition(data, seed, **settings) for seed in seeds]
        results[name] = {"settings": settings, "summary": _summary(runs), "runs": runs}
    return {
        "settings": {
            "architecture": "561→64→32→6",
            "seeds": list(seeds),
            "total_updates": UPDATES,
            "detour_ascent_updates": ASCENT_UPDATES,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "optimizer": "Adam; ascent negates the cross-entropy gradient",
            "split": "strict train/validation/test",
        },
        "conditions": results,
    }


def write_report(result, path):
    lines = ["# 오차 최소화만 vs 오차 증가 후 감소", "", "두 조건은 같은 초기 가중치, 데이터 분할, seed, 총 20회 Adam 업데이트를 사용했다. 기준군은 20회 모두 cross-entropy를 줄이고, 우회군은 10회 gradient ascent로 오차를 키운 뒤 10회 gradient descent로 오차를 줄였다.", ""]
    for name, condition in result["conditions"].items():
        start, peak, end = condition["summary"][0], condition["summary"][10], condition["summary"][-1]
        lines += [f"## {name}", f"- train loss: {start['train_loss']:.3f} → {peak['train_loss']:.3f} → {end['train_loss']:.3f}", f"- validation loss: {start['validation_loss']:.3f} → {peak['validation_loss']:.3f} → {end['validation_loss']:.3f}", f"- test loss: {start['test_loss']:.3f} → {peak['test_loss']:.3f} → {end['test_loss']:.3f}", f"- test accuracy: {start['test_accuracy']:.3f} → {peak['test_accuracy']:.3f} → {end['test_accuracy']:.3f}", ""]
    lines += ["## 해석 기준", "", "오차를 줄이는 방향은 현재 데이터의 정답 확률을 높이는 방향이다. 오차를 키우는 단계는 같은 목적함수의 반대 방향이므로, 특별한 탐색 효과가 없다면 제한된 업데이트 예산에서 기준군보다 불리하거나 회복 시간이 필요할 것으로 예상한다. 최종 비교는 test를 학습에 사용하지 않고 마지막 update의 test loss/accuracy로 판단한다."]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    root = Path(__file__).parent
    result = run_error_detour(root / "UCI HAR Dataset")
    (root / "error_detour_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_report(result, root / "error_detour_analysis.md")
    print(json.dumps({name: condition["summary"][-1] for name, condition in result["conditions"].items()}, indent=2))
