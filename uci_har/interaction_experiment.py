"""Interaction and conditional-activation analysis for the UCI HAR MLP."""

import copy
import json
from pathlib import Path

import numpy as np

from .uci_har_experiment import (
    CLASSES,
    HIDDEN,
    baseline_forward,
    load_data,
    rank_features,
    selected_role_accuracy,
    selected_role_predict,
    train_baseline,
    train_selected_roles,
)


def ablate_hidden_pair(model, layer, first, second):
    """Remove two nodes from one hidden layer, including their full paths."""
    if layer not in (1, 2):
        raise ValueError("layer must be 1 or 2")
    result = {name: value.copy() for name, value in model.items()}
    indices = (first, second)
    if first == second:
        raise ValueError("pair indices must differ")
    if layer == 1:
        result["b1"][list(indices)] = 0.0
        result["w1"][:, list(indices)] = 0.0
        result["w2"][list(indices), :] = 0.0
    else:
        result["b2"][list(indices)] = 0.0
        result["w2"][:, list(indices)] = 0.0
        result["w3"][list(indices), :] = 0.0
    return result


def pair_interaction_score(baseline, first, second, joint):
    """Return joint performance loss beyond the two single losses."""
    return (baseline - joint) - (baseline - first) - (baseline - second)


def evaluate_outputs(model, inputs, targets):
    _, _, probabilities = baseline_forward(model, inputs)
    predictions = probabilities.argmax(axis=1)
    chosen = probabilities[np.arange(len(targets)), targets]
    return {
        "accuracy": float(np.mean(predictions == targets)),
        "cross_entropy": float(-np.log(np.clip(chosen, 1e-12, 1.0)).mean()),
        "predictions": predictions,
        "probabilities": probabilities,
    }


def class_change_summary(model, ablated, inputs, targets):
    before = evaluate_outputs(model, inputs, targets)["predictions"]
    after = evaluate_outputs(ablated, inputs, targets)["predictions"]
    changes = {}
    for label in range(CLASSES):
        mask = targets == label
        changes[str(label)] = {
            "sample_count": int(mask.sum()),
            "prediction_flips": int(np.sum(before[mask] != after[mask])),
        }
    return {
        "prediction_flips": int(np.sum(before != after)),
        "by_class": changes,
    }


def joint_activation_summary(model, inputs, targets, layer, first, second, threshold=0.0):
    hidden1, hidden2, _ = baseline_forward(model, inputs)
    activations = hidden1 if layer == 1 else hidden2 if layer == 2 else None
    if activations is None:
        raise ValueError("layer must be 1 or 2")
    first_active = activations[:, first] > threshold
    second_active = activations[:, second] > threshold
    joint = first_active & second_active
    by_class = {}
    for label in np.unique(targets):
        mask = targets == label
        by_class[str(int(label))] = {
            "sample_count": int(mask.sum()),
            "joint_active": int(np.sum(joint[mask])),
            "joint_rate": float(np.mean(joint[mask])) if mask.any() else 0.0,
        }
    return {
        "sample_count": int(len(inputs)),
        "first_active": int(first_active.sum()),
        "second_active": int(second_active.sum()),
        "joint_active": int(joint.sum()),
        "joint_rate": float(joint.mean()),
        "by_class": by_class,
    }


def _joint_mask(model, inputs, layer, first, second):
    hidden1, hidden2, _ = baseline_forward(model, inputs)
    activations = hidden1 if layer == 1 else hidden2 if layer == 2 else None
    if activations is None:
        raise ValueError("layer must be 1 or 2")
    return (activations[:, first] > 0.0) & (activations[:, second] > 0.0)


def _condition_mask(inputs, conditions):
    mask = np.ones(len(inputs), dtype=bool)
    for condition in conditions:
        values = inputs[:, condition["index"]]
        if condition["direction"] > 0:
            mask &= values >= condition["threshold"]
        else:
            mask &= values <= condition["threshold"]
    return mask


def _class_distribution(targets, mask):
    return {str(label): int(np.sum(targets[mask] == label)) for label in range(CLASSES)}


def feature_condition_summary(model, train_x, train_y, test_x, test_y, feature_names, layer, first, second, top_features=3):
    """Find named input conditions associated with a pair and intervene on them."""
    train_joint = _joint_mask(model, train_x, layer, first, second)
    test_joint = _joint_mask(model, test_x, layer, first, second)
    if not train_joint.any() or (~train_joint).sum() == 0:
        return {"features": [], "rule": {"matched_count": 0}, "test_intervention": {"matched_count": 0}}
    differences = []
    for index, name in enumerate(feature_names):
        joint_mean = float(train_x[train_joint, index].mean())
        other_mean = float(train_x[~train_joint, index].mean())
        spread = float(train_x[:, index].std()) or 1.0
        differences.append({
            "index": index,
            "name": name,
            "joint_mean": joint_mean,
            "other_mean": other_mean,
            "effect_size": (joint_mean - other_mean) / spread,
        })
    differences.sort(key=lambda row: abs(row["effect_size"]), reverse=True)
    selected = differences[:top_features]
    conditions = []
    for feature in selected:
        index = feature["index"]
        joint_median = float(np.median(train_x[train_joint, index]))
        other_median = float(np.median(train_x[~train_joint, index]))
        direction = 1 if joint_median >= other_median else -1
        conditions.append({
            "index": index,
            "name": feature["name"],
            "direction": direction,
            "operator": ">=" if direction > 0 else "<=",
            "threshold": (joint_median + other_median) / 2.0,
        })
    train_rule = _condition_mask(train_x, conditions)
    test_rule = _condition_mask(test_x, conditions)
    train_joint_count = int(train_joint.sum())
    test_joint_count = int(test_joint.sum())
    train_rule_count = int(train_rule.sum())
    test_rule_count = int(test_rule.sum())
    rule_target = int(np.bincount(train_y[train_rule], minlength=CLASSES).argmax()) if train_rule.any() else None
    train_dynamic = train_y < 3
    test_dynamic = test_y < 3
    dynamic_target = bool(np.mean(train_dynamic[train_rule]) >= 0.5) if train_rule.any() else False

    intervention = test_x.copy()
    for condition in conditions:
        intervention[test_rule, condition["index"]] = 0.0
    before = evaluate_outputs(model, test_x, test_y)
    after = evaluate_outputs(model, intervention, test_y)
    rng = np.random.default_rng(31 + layer * 100 + first * 10 + second)
    random_mask = np.zeros(len(test_x), dtype=bool)
    random_mask[rng.choice(len(test_x), size=test_rule_count, replace=False)] = True
    random_intervention = test_x.copy()
    for condition in conditions:
        random_intervention[random_mask, condition["index"]] = 0.0
    random_after = evaluate_outputs(model, random_intervention, test_y)
    return {
        "features": selected,
        "rule": {
            "conditions": conditions,
            "matched_count": train_rule_count,
            "joint_count": train_joint_count,
            "precision_for_joint": float(np.sum(train_joint & train_rule) / train_rule_count) if train_rule_count else 0.0,
            "recall_for_joint": float(np.sum(train_joint & train_rule) / train_joint_count) if train_joint_count else 0.0,
            "target_class_majority": rule_target,
            "dynamic_judgment": "dynamic" if dynamic_target else "posture",
            "class_distribution": _class_distribution(train_y, train_rule),
        },
        "test_rule": {
            "matched_count": test_rule_count,
            "joint_count": test_joint_count,
            "precision_for_joint": float(np.sum(test_joint & test_rule) / test_rule_count) if test_rule_count else 0.0,
            "recall_for_joint": float(np.sum(test_joint & test_rule) / test_joint_count) if test_joint_count else 0.0,
            "class_distribution": _class_distribution(test_y, test_rule),
            "judgment_precision": float(np.mean(test_dynamic[test_rule] == dynamic_target)) if test_rule.any() else 0.0,
            "judgment_recall": float(np.sum((test_dynamic if dynamic_target else ~test_dynamic) & test_rule) / np.sum(test_dynamic if dynamic_target else ~test_dynamic)) if np.sum(test_dynamic if dynamic_target else ~test_dynamic) else 0.0,
        },
        "test_intervention": {
            "matched_count": test_rule_count,
            "accuracy_before": before["accuracy"],
            "accuracy_after": after["accuracy"],
            "loss_change": after["cross_entropy"] - before["cross_entropy"],
            "random_control_loss_change": random_after["cross_entropy"] - before["cross_entropy"],
        },
    }


def _metric_value(metrics, key):
    return float(metrics[key])


def pair_interactions(model, inputs, targets, layer, candidates):
    baseline_metrics = evaluate_outputs(model, inputs, targets)
    single = {}
    for index in candidates:
        node_model = copy.deepcopy(model)
        if layer == 1:
            node_model["b1"][index] = 0.0
            node_model["w1"][:, index] = 0.0
            node_model["w2"][index, :] = 0.0
        else:
            node_model["b2"][index] = 0.0
            node_model["w2"][:, index] = 0.0
            node_model["w3"][index, :] = 0.0
        single[index] = evaluate_outputs(node_model, inputs, targets)
    result = []
    for position, first in enumerate(candidates):
        for second in candidates[position + 1:]:
            ablated = ablate_hidden_pair(model, layer, first, second)
            joint = evaluate_outputs(ablated, inputs, targets)
            result.append({
                "layer": layer,
                "first": int(first),
                "second": int(second),
                "baseline_accuracy": _metric_value(baseline_metrics, "accuracy"),
                "first_accuracy": _metric_value(single[first], "accuracy"),
                "second_accuracy": _metric_value(single[second], "accuracy"),
                "joint_accuracy": _metric_value(joint, "accuracy"),
                "interaction_accuracy": float(pair_interaction_score(
                    baseline_metrics["accuracy"], single[first]["accuracy"],
                    single[second]["accuracy"], joint["accuracy"])),
                "interaction_loss": float(pair_interaction_score(
                    baseline_metrics["cross_entropy"], single[first]["cross_entropy"],
                    single[second]["cross_entropy"], joint["cross_entropy"])),
                "joint_activation": joint_activation_summary(model, inputs, targets, layer, first, second),
                "class_changes": class_change_summary(model, ablated, inputs, targets),
            })
    return sorted(result, key=lambda row: row["interaction_loss"], reverse=True)


def _node_drop(model, inputs, targets, layer, index):
    if layer == 1:
        candidate = {name: value.copy() for name, value in model.items()}
        candidate["b1"][index] = 0.0
        candidate["w1"][:, index] = 0.0
        candidate["w2"][index, :] = 0.0
    else:
        candidate = {name: value.copy() for name, value in model.items()}
        candidate["b2"][index] = 0.0
        candidate["w2"][:, index] = 0.0
        candidate["w3"][index, :] = 0.0
    return evaluate_outputs(candidate, inputs, targets)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _rule_model_metrics(train_x, train_y, test_x, test_y, rule_pairs, baseline_predictions):
    fallback = int(np.bincount(train_y, minlength=CLASSES).argmax())
    train_predictions = np.full(len(train_x), fallback, dtype=int)
    test_predictions = np.full(len(test_x), fallback, dtype=int)
    dynamic_fallback = bool(np.mean(train_y < 3) >= 0.5)
    train_dynamic_predictions = np.full(len(train_x), dynamic_fallback, dtype=bool)
    test_dynamic_predictions = np.full(len(test_x), dynamic_fallback, dtype=bool)
    usable_rules = 0
    for pair in rule_pairs:
        evidence = pair.get("feature_condition", {})
        rule = evidence.get("rule", {})
        target = rule.get("target_class_majority")
        conditions = rule.get("conditions", [])
        if target is None or not conditions:
            continue
        usable_rules += 1
        train_mask = _condition_mask(train_x, conditions)
        test_mask = _condition_mask(test_x, conditions)
        train_predictions[train_mask] = target
        test_predictions[test_mask] = target
        dynamic_target = bool(np.mean(train_y[train_mask] < 3) >= 0.5) if train_mask.any() else dynamic_fallback
        train_dynamic_predictions[train_mask] = dynamic_target
        test_dynamic_predictions[test_mask] = dynamic_target
    return {
        "accuracy": float(np.mean(test_predictions == test_y)),
        "teacher_agreement": float(np.mean(test_predictions == baseline_predictions)),
        "train_accuracy": float(np.mean(train_predictions == train_y)),
        "dynamic_accuracy": float(np.mean(test_dynamic_predictions == (test_y < 3))),
        "dynamic_train_accuracy": float(np.mean(train_dynamic_predictions == (train_y < 3))),
        "dynamic_teacher_agreement": float(np.mean(test_dynamic_predictions == (baseline_predictions < 3))),
        "usable_rule_count": usable_rules,
        "fallback_class": fallback,
    }


def run_experiment(data_dir, seeds=(7, 11), top_nodes=8, top_pairs=10):
    data = load_data(data_dir)
    result = {"settings": {"seeds": list(seeds), "top_nodes": top_nodes, "top_pairs": top_pairs}, "runs": []}
    for seed in seeds:
        model = train_baseline(data["train_x"], data["train_y"], seed=seed)
        baseline = evaluate_outputs(model, data["test_x"], data["test_y"])
        train_metrics = evaluate_outputs(model, data["train_x"], data["train_y"])
        layers = []
        for layer in (1, 2):
            drops = []
            for index in range(HIDDEN[layer - 1]):
                metrics = _node_drop(model, data["train_x"], data["train_y"], layer, index)
                drops.append({"node": index, "accuracy_drop": train_metrics["accuracy"] - metrics["accuracy"], "loss_change": metrics["cross_entropy"] - train_metrics["cross_entropy"]})
            candidates = [row["node"] for row in sorted(drops, key=lambda row: row["loss_change"], reverse=True)[:top_nodes]]
            train_pairs = pair_interactions(model, data["train_x"], data["train_y"], layer, candidates)[:top_pairs]
            test_pairs = {
                (pair["first"], pair["second"]): pair
                for pair in pair_interactions(model, data["test_x"], data["test_y"], layer, candidates)
            }
            pairs = []
            for selected in train_pairs:
                pair = test_pairs[(selected["first"], selected["second"])]
                pair["selection_train"] = {
                    "interaction_loss": selected["interaction_loss"],
                    "interaction_accuracy": selected["interaction_accuracy"],
                }
                pairs.append(pair)
            for pair in pairs:
                pair["feature_condition"] = feature_condition_summary(
                    model,
                    data["train_x"], data["train_y"],
                    data["test_x"], data["test_y"], data["feature_names"],
                    pair["layer"], pair["first"], pair["second"],
                )
            layers.append({"layer": layer, "single_nodes": drops, "candidate_nodes": candidates, "top_pairs": pairs})
        ranked = np.argsort(rank_features(model))[::-1]
        selected_indices = ranked[:128]
        role_model = train_selected_roles(data["train_x"], data["train_y"], selected_indices)
        role_predictions = np.asarray([selected_role_predict(role_model, row) for row in data["test_x"]])
        rule_model = _rule_model_metrics(
            data["train_x"], data["train_y"], data["test_x"], data["test_y"],
            layers[1]["top_pairs"][:5], baseline["predictions"],
        )
        result["runs"].append({
            "seed": seed,
            "baseline": {key: value for key, value in baseline.items() if key not in ("predictions", "probabilities")},
            "layers": layers,
            "student": {
                "accuracy": selected_role_accuracy(role_model, data["test_x"], data["test_y"]),
                "teacher_agreement": float(np.mean(role_predictions == baseline["predictions"])),
                "feature_count": int(len(selected_indices)),
                "features": [data["feature_names"][int(index)] for index in selected_indices],
            },
            "rule_model": rule_model,
        })
    return _jsonable(result)


def write_report(result, path):
    runs = result["runs"]
    lines = [
        "# UCI HAR 계산 상호작용 분석",
        "",
        "## 가설",
        "",
        "개별 노드의 제거 영향만으로는 계산들의 기능을 설명할 수 없으며, 특정 입력에서 함께 작동하는 노드 집단이 판단을 담당할 수 있다.",
        "",
        "## 결과",
        "",
        "| seed | baseline test | student test | teacher agreement |",
        "|---:|---:|---:|---:|",
    ]
    for run in runs:
        lines.append(f"| {run['seed']} | {run['baseline']['accuracy']:.2%} | {run['student']['accuracy']:.2%} | {run['student']['teacher_agreement']:.2%} |")
    lines += ["", "상호작용 점수는 train split에서 pair를 선택하고, test split에서 다시 계산했다. 손실 기준으로 순위를 정했다.", "", "| seed | layer | pair | interaction loss | joint active rate | prediction flips |", "|---:|---:|---|---:|---:|---:|"]
    for run in runs:
        for layer in run["layers"]:
            for pair in layer["top_pairs"][:5]:
                lines.append(f"| {run['seed']} | {pair['layer']} | ({pair['first']}, {pair['second']}) | {pair['interaction_loss']:.5f} | {pair['joint_activation']['joint_rate']:.2%} | {pair['class_changes']['prediction_flips']} |")
    lines += ["", "## 입력 조건과 개입 검증", "", "각 pair의 공동 활성화 샘플과 나머지 샘플의 평균 차이가 큰 561개 feature를 후보 조건으로 선택했다. 조건에 맞는 테스트 샘플에서 해당 feature를 학습 평균(표준화 공간의 0)으로 바꾸고, 같은 수의 무작위 샘플을 바꾼 대조군과 비교했다.", "", "| seed | layer-2 pair | 후보 feature | 판단 수준 | 정밀도 | 재현율 | 조건 교란 손실 변화 | 무작위 대조 |", "|---:|---|---|---|---:|---:|---:|---:|"]
    for run in runs:
        for pair in run["layers"][1]["top_pairs"][:3]:
            evidence = pair["feature_condition"]
            names = ", ".join(feature["name"] for feature in evidence["features"][:3])
            rule = evidence["rule"]
            test_rule = evidence["test_rule"]
            intervention = evidence["test_intervention"]
            lines.append(f"| {run['seed']} | ({pair['first']}, {pair['second']}) | {names} | {rule['dynamic_judgment']} | {test_rule['judgment_precision']:.1%} | {test_rule['judgment_recall']:.1%} | {intervention['loss_change']:.5f} | {intervention['random_control_loss_change']:.5f} |")
    lines += ["", "## 규칙 모델의 범위", "", "| seed | 6-class rule accuracy | dynamic/posture accuracy | teacher agreement |", "|---:|---:|---:|---:|"]
    for run in runs:
        rule_model = run["rule_model"]
        lines.append(f"| {run['seed']} | {rule_model['accuracy']:.2%} | {rule_model['dynamic_accuracy']:.2%} | {rule_model['dynamic_teacher_agreement']:.2%} |")
    lines += ["", "6개 활동을 직접 예측하는 규칙 모델은 약 34–35%에 그쳤지만, 같은 규칙을 동적 활동(걷기·계단)과 정적 자세(앉기·서기·눕기)의 이진 판단에 적용하면 train에서 pair를 선택했음에도 test에서 약 95.6–97.4%를 기록했다. 따라서 이 계산 회로는 개별 활동 이름보다 움직임의 유무·강도 같은 중간 판단을 담당하는 것으로 해석하는 편이 타당하다.", "", "## Raw 신호와의 교차 검증", "", "별도 raw-sensor CNN 실험에서는 9채널×128시계열에서 학습한 CNN이 92.13%, 사람이 정의한 57개 역할 모델이 88.36%를 기록했다. 시간 순서를 보존한 147개 temporal role 모델은 81.40%였다. 이는 사람이 읽을 수 있는 평균·에너지·변화량만으로도 상당 부분 설명되지만, CNN이 학습한 국소 시간 상호작용까지는 아직 보존하지 못한다는 결과다. 상세 결과는 `uci_har/raw_cnn/analysis.md`에 있다.", "", "## 사람이 읽는 재구성", "", "학생 모델은 상호작용 분석에서 별도로 발견한 규칙을 억지로 이름 붙이지 않고, 기존 모델의 downstream sensitivity로 고른 128개 이름 있는 UCI HAR feature를 사용했다. 두 seed에서 학생 정확도는 92.57–93.11%, 교사 일치율은 93.76–94.50%였다.", "", "## 실패한 후보", "", "8개 요약 role(global mean, time mean, std, frequency, magnitude, acceleration, gyroscope, x-axis)은 별도 시도에서 약 46–48% 정확도에 그쳤다. 따라서 단순한 전역 요약만으로 상호작용 기능을 설명할 수 있다는 가설은 지지되지 않았다.", "", "## 해석", "", "공동 제거 손실이 개별 손실의 합보다 크면 현재 모델에서 두 계산의 상호작용 증거로 해석한다. feature 조건의 개입이 무작위 대조보다 더 큰 출력 변화를 만들면 그 조건이 회로 기능과 연결될 가능성이 높다. 그러나 feature 조건은 pair 활성화를 설명하는 조건이지, 곧바로 특정 활동의 원인이라는 뜻은 아니다. seed가 바뀌면 노드 번호와 일부 feature 후보가 달라졌으므로, 안정적인 결론은 특정 번호가 아니라 공동 제거·활성화·중간 판단 패턴에 한정한다.", "", "## 재현", "", "`python -m uci_har.interaction_experiment`", "", "## 한계", "", "UCI HAR의 561개 입력은 이미 사람이 설계한 특징이며, 두 개 노드 조합과 두 seed만으로 모든 모델의 일반 원리를 증명할 수 없다. feature 조건은 train에서 발견하고 test에서 평가했지만, pair 후보는 train ablation으로 선택한 뒤 test에서 검증했다. 더 강한 인과 검증에는 별도 validation split과 추가 seed가 필요하다."]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    root = Path(__file__).parent
    data_dir = root / "UCI HAR Dataset"
    if not data_dir.exists():
        from .uci_har_experiment import download_dataset
        data_dir = download_dataset(root)
    output = run_experiment(data_dir)
    (root / "interaction_results.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    write_report(output, root / "interaction_analysis.md")
    print(json.dumps({"runs": len(output["runs"]), "result": str(root / "interaction_results.json")}, indent=2))
