"""Hierarchical activity analysis built on the existing interaction experiment."""

import json
from pathlib import Path

import numpy as np

from .interaction_experiment import (
    _condition_mask,
    _joint_mask,
    _node_drop,
    ablate_hidden_pair,
    evaluate_outputs,
    feature_condition_summary,
    pair_interactions,
)
from .uci_har_experiment import CLASSES, HIDDEN, load_data, train_baseline


DYNAMIC = np.array([0, 1, 2])
STATIC = np.array([3, 4, 5])


def classwise_metrics(model, inputs, targets):
    metrics = evaluate_outputs(model, inputs, targets)
    predictions = metrics["predictions"]
    label_count = max(CLASSES, int(np.max(targets)) + 1 if len(targets) else CLASSES)
    confusion = [[0 for _ in range(label_count)] for _ in range(label_count)]
    for target, prediction in zip(targets, predictions):
        confusion[int(target)][int(prediction)] += 1
    by_class = {}
    for label in range(label_count):
        mask = targets == label
        by_class[str(label)] = {
            "count": int(mask.sum()),
            "accuracy": float(np.mean(predictions[mask] == label)) if mask.any() else None,
        }
    return {
        "accuracy": metrics["accuracy"],
        "cross_entropy": metrics["cross_entropy"],
        "sample_count": int(len(targets)),
        "by_class": by_class,
        "confusion": confusion,
    }


def ablation_effect_by_class(model, ablated, inputs, targets):
    baseline = classwise_metrics(model, inputs, targets)
    changed = classwise_metrics(ablated, inputs, targets)
    by_class = {}
    for label, before in baseline["by_class"].items():
        after = changed["by_class"][label]
        by_class[label] = {
            "baseline_accuracy": before["accuracy"],
            "ablated_accuracy": after["accuracy"],
            "accuracy_delta": (after["accuracy"] - before["accuracy"])
            if before["accuracy"] is not None and after["accuracy"] is not None else None,
        }
    return {"baseline": baseline, "ablated": changed, "by_class": by_class}


def hierarchy_relation(dynamic_pair, detail_pair):
    dynamic_features = set(dynamic_pair.get("feature_names", []))
    detail_features = set(detail_pair.get("feature_names", []))
    return {
        "downstream_layer_order": detail_pair["layer"] > dynamic_pair["layer"],
        "same_layer": detail_pair["layer"] == dynamic_pair["layer"],
        "feature_overlap": sorted(dynamic_features & detail_features),
        "dynamic_pair": [dynamic_pair["layer"], dynamic_pair["first"], dynamic_pair["second"]],
        "detail_pair": [detail_pair["layer"], detail_pair["first"], detail_pair["second"]],
    }


def _validated_pairs(model, train_x, train_y, test_x, test_y, feature_names, layer, top_nodes=8, top_pairs=5):
    train_baseline = evaluate_outputs(model, train_x, train_y)
    drops = []
    for index in range(HIDDEN[layer - 1]):
        metrics = _node_drop(model, train_x, train_y, layer, index)
        drops.append({"node": index, "loss_change": metrics["cross_entropy"] - train_baseline["cross_entropy"]})
    candidates = [row["node"] for row in sorted(drops, key=lambda row: row["loss_change"], reverse=True)[:top_nodes]]
    train_pairs = pair_interactions(model, train_x, train_y, layer, candidates)[:top_pairs]
    test_pairs = {(row["first"], row["second"]): row for row in pair_interactions(model, test_x, test_y, layer, candidates)}
    selected = []
    for train_pair in train_pairs:
        key = (train_pair["first"], train_pair["second"])
        test_pair = test_pairs[key]
        test_pair["selection_train"] = {
            "interaction_loss": train_pair["interaction_loss"],
            "interaction_accuracy": train_pair["interaction_accuracy"],
        }
        evidence = feature_condition_summary(
            model, train_x, train_y, test_x, test_y, feature_names,
            layer, test_pair["first"], test_pair["second"],
        )
        test_pair["feature_condition"] = evidence
        test_pair["feature_names"] = [row["name"] for row in evidence["features"]]
        selected.append(test_pair)
    return {"layer": layer, "candidate_nodes": candidates, "pairs": selected}


def _pick_pair(layer_result, judgment=None):
    if judgment is None:
        return layer_result["pairs"][0]
    matches = [pair for pair in layer_result["pairs"] if pair["feature_condition"]["rule"].get("dynamic_judgment") == judgment]
    return matches[0] if matches else layer_result["pairs"][0]


def _stage(name, hypothesis, method, expected, actual, mismatch, next_question):
    return {
        "name": name,
        "hypothesis": hypothesis,
        "method": method,
        "expected": expected,
        "actual": actual,
        "mismatch": mismatch,
        "next_question": next_question,
    }


def _hierarchical_rule_metrics(model, dynamic_pair, detail_pair, train_x, train_y, test_x, test_y):
    dynamic_conditions = dynamic_pair["feature_condition"]["rule"]["conditions"]
    detail_conditions = detail_pair["feature_condition"]["rule"]["conditions"]
    train_dynamic_rule = _condition_mask(train_x, dynamic_conditions)
    test_dynamic_rule = _condition_mask(test_x, dynamic_conditions)
    train_detail_rule = _condition_mask(train_x, detail_conditions)
    test_detail_rule = _condition_mask(test_x, detail_conditions)
    static_train = train_y >= 3
    static_fallback = int(np.bincount(train_y[static_train], minlength=CLASSES).argmax()) if static_train.any() else 3
    dynamic_train = train_y < 3
    dynamic_fallback = int(np.bincount(train_y[dynamic_train], minlength=CLASSES).argmax()) if dynamic_train.any() else 0
    detail_train = train_y[train_dynamic_rule & train_detail_rule]
    detail_fallback = int(np.bincount(detail_train, minlength=CLASSES).argmax()) if len(detail_train) else dynamic_fallback
    predictions = np.full(len(test_y), static_fallback, dtype=int)
    dynamic_rows = test_dynamic_rule
    predictions[dynamic_rows] = detail_fallback
    predictions[dynamic_rows & test_detail_rule] = detail_fallback
    return {
        "accuracy": float(np.mean(predictions == test_y)),
        "teacher_agreement": float(np.mean(predictions == evaluate_outputs(model, test_x, test_y)["predictions"])),
        "dynamic_accuracy": float(np.mean((predictions < 3) == (test_y < 3))),
        "dynamic_subset_accuracy": float(np.mean(predictions[test_y < 3] == test_y[test_y < 3])),
        "fallback_static_class": static_fallback,
        "fallback_dynamic_class": dynamic_fallback,
        "detail_rule_class": detail_fallback,
        "dynamic_rule_matches": int(test_dynamic_rule.sum()),
        "detail_rule_matches": int(test_detail_rule.sum()),
    }


def run_hierarchical_experiment(data_dir, seeds=(7, 11)):
    data = load_data(data_dir)
    result = {"settings": {"seeds": list(seeds), "dynamic_labels": DYNAMIC.tolist(), "static_labels": STATIC.tolist()}, "runs": []}
    for seed in seeds:
        model = train_baseline(data["train_x"], data["train_y"], seed=seed)
        all_layers = []
        for layer in (1, 2):
            all_layers.append(_validated_pairs(
                model, data["train_x"], data["train_y"], data["test_x"], data["test_y"], data["feature_names"], layer,
            ))
        dynamic_train_mask = np.isin(data["train_y"], DYNAMIC)
        dynamic_test_mask = np.isin(data["test_y"], DYNAMIC)
        detail_layers = []
        for layer in (1, 2):
            detail_layers.append(_validated_pairs(
                model,
                data["train_x"][dynamic_train_mask], data["train_y"][dynamic_train_mask],
                data["test_x"][dynamic_test_mask], data["test_y"][dynamic_test_mask], data["feature_names"], layer,
            ))
        dynamic_pair = _pick_pair(all_layers[1], "dynamic")
        detail_pair = _pick_pair(detail_layers[1])
        dynamic_ablation = ablation_effect_by_class(
            model,
            ablate_hidden_pair(model, dynamic_pair["layer"], dynamic_pair["first"], dynamic_pair["second"]),
            data["test_x"], data["test_y"],
        )
        detail_ablation = ablation_effect_by_class(
            model,
            ablate_hidden_pair(model, detail_pair["layer"], detail_pair["first"], detail_pair["second"]),
            data["test_x"][dynamic_test_mask], data["test_y"][dynamic_test_mask],
        )
        hierarchy = hierarchy_relation(dynamic_pair, detail_pair)
        hierarchy["test_detail_given_dynamic"] = float(
            (_joint_mask(model, data["test_x"], detail_pair["layer"], detail_pair["first"], detail_pair["second"]) &
             _joint_mask(model, data["test_x"], dynamic_pair["layer"], dynamic_pair["first"], dynamic_pair["second"])).sum() /
            max(1, _joint_mask(model, data["test_x"], dynamic_pair["layer"], dynamic_pair["first"], dynamic_pair["second"]).sum())
        )
        rule_model = _hierarchical_rule_metrics(
            model, dynamic_pair, detail_pair,
            data["train_x"], data["train_y"], data["test_x"], data["test_y"],
        )
        dynamic_deltas = [row["accuracy_delta"] for label, row in dynamic_ablation["by_class"].items() if int(label) in DYNAMIC and row["accuracy_delta"] is not None]
        static_deltas = [row["accuracy_delta"] for label, row in dynamic_ablation["by_class"].items() if int(label) in STATIC and row["accuracy_delta"] is not None]
        result["runs"].append({
            "seed": seed,
            "baseline": classwise_metrics(model, data["test_x"], data["test_y"]),
            "dynamic_static_pair": dynamic_pair,
            "dynamic_static_ablation": dynamic_ablation,
            "dynamic_subset_layers": detail_layers,
            "detail_pair": detail_pair,
            "detail_ablation_dynamic_subset": detail_ablation,
            "hierarchy": hierarchy,
            "rule_model": rule_model,
            "stages": [
                _stage("dynamic_static_ablation", "중간 판단 pair는 6개 class에 공통 영향을 준다.", "test에서 pair 제거 후 class별 정확도와 confusion 비교", "비슷한 정확도 저하", dynamic_deltas + static_deltas, "class별 저하가 다르면 공통 회로 가설이 약해진다." if max(dynamic_deltas + static_deltas) - min(dynamic_deltas + static_deltas) > 0.05 else "큰 모순 없음", "pair가 특정 class 또는 동적/정적 경계에 편향되는가?"),
                _stage("dynamic_subset", "동적 3개 class에는 별도의 pair가 있다.", "동적 train subset에서 pair 선택, 동적 test subset에서 검증", "새 pair와 feature 조건", {"layer": detail_pair["layer"], "pair": [detail_pair["first"], detail_pair["second"]]}, "기존 pair와 같으면 별도 세부 회로 가설이 약해진다." if [detail_pair["first"], detail_pair["second"]] == [dynamic_pair["first"], dynamic_pair["second"]] else "새 pair 관찰", "새 pair가 세 동적 활동 중 무엇을 구분하는가?"),
                _stage("feature_intervention", "새 pair feature를 교란하면 무작위보다 출력이 더 변한다.", "조건 feature를 0으로 교란하고 동일 크기 무작위 대조", "조건 교란 손실 변화가 대조군보다 큼", detail_pair["feature_condition"]["test_intervention"], "대조군보다 작거나 음수면 feature 조건의 인과 해석이 약해진다." if detail_pair["feature_condition"]["test_intervention"]["loss_change"] <= detail_pair["feature_condition"]["test_intervention"]["random_control_loss_change"] else "조건 교란이 대조군보다 큼", "더 분리된 validation split에서도 유지되는가?"),
                _stage("hierarchy", "세부 pair는 동적/정적 pair의 downstream 계산이다.", "layer 순서, feature 중복, 공동 활성화 조건부 확률 비교", "downstream layer와 조건부 활성화/feature 연결", hierarchy, "구조적 downstream이 아니거나 overlap이 없으면 계층 가설은 미확정이다." if not hierarchy["downstream_layer_order"] or not hierarchy["feature_overlap"] else "일부 계층 증거", "같은 layer에서 기능적 계층을 어떻게 정의할 것인가?"),
                _stage("hierarchical_rule", "센서 feature→중간 판단→세부 활동 규칙으로 원래 판단을 근사할 수 있다.", "dynamic rule 뒤 detail rule을 적용한 end-to-end 평가", "6-class와 동적 subset 모두 높은 일치", rule_model, "6-class 정확도가 낮으면 계층 구조가 완전한 설명은 아니다." if rule_model["accuracy"] < 0.7 else "규칙 구조가 유효함", "세부 활동을 구분하는 더 풍부한 temporal feature가 필요한가?"),
            ],
        })
    return result


def write_hierarchical_report(result, path):
    lines = [
        "# 계층적 활동 판단 분석",
        "",
        "이 보고서는 센서 feature → 동적/정적 중간 판단 → 세부 활동 판단 가설을 단계별로 검증한다.",
        "",
    ]
    for run in result["runs"]:
        lines += [f"## Seed {run['seed']}", ""]
        for stage in run["stages"]:
            lines += [f"### {stage['name']}", "", f"- 가설: {stage['hypothesis']}", f"- 방법: {stage['method']}", f"- 예상: {stage['expected']}", f"- 실제: `{json.dumps(stage['actual'], ensure_ascii=False)}`", f"- 불일치: {stage['mismatch']}", f"- 다음 질문: {stage['next_question']}", ""]
        lines += ["### 주요 pair와 feature", "", f"- 동적/정적 pair: layer {run['dynamic_static_pair']['layer']} ({run['dynamic_static_pair']['first']}, {run['dynamic_static_pair']['second']})", f"- 세부 동적 pair: layer {run['detail_pair']['layer']} ({run['detail_pair']['first']}, {run['detail_pair']['second']})", f"- 동적/정적 ablation class 결과: `{json.dumps(run['dynamic_static_ablation']['by_class'], ensure_ascii=False)}`", f"- 세부 pair feature: {', '.join(run['detail_pair']['feature_names'])}", ""]
    lines += ["## 종합 결론", "", "현재 증거는 동적/정적 중간 판단과 동적 3개 활동 내부의 추가 계산이 존재할 가능성을 지지한다. 그러나 계층 규칙만으로 6개 활동 전체를 높은 정확도로 재현하지 못하면, 이는 완성된 해석이 아니라 부분적인 기능 분해다. 특히 pair가 다른 layer의 downstream 계산인지, 같은 layer의 병렬 계산인지 구분해야 한다.", "", "## 재현", "", "`python -m uci_har.hierarchical_experiment`"]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    root = Path(__file__).parent
    data_dir = root / "UCI HAR Dataset"
    output = run_hierarchical_experiment(data_dir)
    (root / "hierarchical_results.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    write_hierarchical_report(output, root / "hierarchical_analysis.md")
    print(json.dumps({"runs": len(output["runs"]), "result": str(root / "hierarchical_results.json")}, indent=2))
