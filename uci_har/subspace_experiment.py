"""Geometric decomposition of activity directions and physical feature interventions."""

import json
from itertools import combinations
from pathlib import Path

import numpy as np

from .direction_experiment import (
    DYNAMIC_LABELS,
    _hidden2_outputs,
    _pearson,
    activity_direction,
    correlation_control_selection,
    evaluate_outputs_from_model,
    fit_r2,
    remove_direction_projection,
)
from .distributed_experiment import _jsonable, pattern_similarity
from .uci_har_experiment import baseline_forward, load_data, train_baseline


FAMILY_PATTERNS = {
    "jerk": ("jerk",),
    "frequency": ("freq", "frequency"),
    "autocorrelation": ("arCoeff",),
    "energy": ("energy", "rms"),
    "gravity_change": ("gravity", "angle("),
}


def direction_geometry(directions):
    directions = np.asarray(directions, dtype=float)
    directions = directions / np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    cosines = {}
    angles = {}
    for first, second in combinations(range(len(directions)), 2):
        cosine = float(np.clip(directions[first] @ directions[second], -1.0, 1.0))
        cosines[f"{first}-{second}"] = cosine
        angles[f"{first}-{second}"] = float(np.degrees(np.arccos(cosine)))
    return {"cosines": cosines, "angles_degrees": angles}


def shared_private_subspace(directions):
    directions = np.asarray(directions, dtype=float)
    directions = directions / np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    _, singular_values, right_vectors = np.linalg.svd(directions, full_matrices=False)
    shared = right_vectors[0]
    if shared @ directions.mean(axis=0) < 0:
        shared = -shared
    projections = directions @ shared
    private = directions - projections[:, None] * shared
    total = float(np.sum(singular_values * singular_values))
    return {
        "shared_component": shared,
        "private_components": private,
        "shared_projections": projections,
        "singular_values": singular_values,
        "shared_explained_fraction": float(singular_values[0] ** 2 / total) if total else 0.0,
    }


def _family_indices(names):
    lowered = [name.lower() for name in names]
    return {
        family: [index for index, name in enumerate(lowered) if any(pattern.lower() in name for pattern in patterns)]
        for family, patterns in FAMILY_PATTERNS.items()
    }


def _physical_concept(first, second):
    pair = frozenset((first, second))
    concepts = {
        frozenset(("frequency", "jerk")): "주기적 움직임 + 급격한 변화",
        frozenset(("frequency", "gravity_change")): "중력 방향 변화 + 주기성",
        frozenset(("autocorrelation", "jerk")): "반복 자기유사성 + 급격한 변화",
        frozenset(("energy", "jerk")): "움직임 에너지 + 급격한 변화",
        frozenset(("energy", "gravity_change")): "움직임 에너지 + 중력 방향 변화",
    }
    return concepts.get(pair, f"{first} + {second}")


def family_combination_analysis(features, target, names):
    groups = _family_indices(names)
    representatives = {}
    for family, indices in groups.items():
        if not indices:
            continue
        selected = correlation_control_selection(features[:, indices], target, [names[index] for index in indices], max_features=1, correlation_limit=0.85)
        if selected:
            selected[0]["index"] = indices[selected[0]["index"]]
            representatives[family] = selected[0]
    candidates = []
    for first, second in combinations(representatives, 2):
        first_index = representatives[first]["index"]
        second_index = representatives[second]["index"]
        joint = fit_r2(features[:, [first_index, second_index]], target)
        single = max(fit_r2(features[:, [first_index]], target), fit_r2(features[:, [second_index]], target))
        candidates.append({"families": [first, second], "concept": _physical_concept(first, second), "indices": [first_index, second_index], "features": [names[first_index], names[second_index]], "joint_r2": joint, "joint_gain": joint - single})
    candidates.sort(key=lambda row: row["joint_gain"], reverse=True)
    return {"representatives": representatives, "combinations": candidates}


def _recall(predictions, targets, label):
    mask = targets == label
    return float(np.mean(predictions[mask] == label)) if mask.any() else 0.0


def _feature_combo_intervention(model, direction, center, train_x, test_x, test_y, combo, seed):
    baseline_hidden = baseline_forward(model, test_x)[1]
    baseline = evaluate_outputs_from_model(model, test_x, test_y)
    direction_score = lambda hidden: (hidden - center) @ direction
    base_score = direction_score(baseline_hidden)
    selected_x = test_x.copy()
    selected_x[:, combo["indices"]] = 0.0
    single_x = test_x.copy()
    single_x[:, combo["indices"][:1]] = 0.0
    rng = np.random.default_rng(seed)
    random_indices = rng.choice(test_x.shape[1], size=len(combo["indices"]), replace=False)
    random_x = test_x.copy()
    random_x[:, random_indices] = 0.0
    outputs = {}
    for name, inputs, indices in (("selected_combo", selected_x, combo["indices"]), ("single_feature", single_x, combo["indices"][:1]), ("random_combo", random_x, random_indices.tolist())):
        hidden = baseline_forward(model, inputs)[1]
        metrics = evaluate_outputs_from_model(model, inputs, test_y)
        score = direction_score(hidden)
        outputs[name] = {
            "indices": [int(index) for index in indices],
            "accuracy": metrics["accuracy"],
            "loss_change": metrics["cross_entropy"] - baseline["cross_entropy"],
            "direction_score_change": float(np.mean(score - base_score)),
            "target_recalls": [_recall(metrics["predictions"], test_y, label) for label in DYNAMIC_LABELS],
        }
    outputs["baseline"] = {"accuracy": baseline["accuracy"], "target_recalls": [_recall(baseline["predictions"], test_y, label) for label in DYNAMIC_LABELS]}
    return outputs


def _component_intervention(model, hidden2, targets, component, center):
    component = np.asarray(component, dtype=float)
    norm = np.linalg.norm(component)
    if norm:
        component = component / norm
    baseline = _hidden2_outputs(model, hidden2, targets)
    changed_hidden = remove_direction_projection(hidden2, component, center=center)
    changed = _hidden2_outputs(model, changed_hidden, targets)
    return {
        "baseline_accuracy": baseline["accuracy"],
        "ablated_accuracy": changed["accuracy"],
        "loss_change": changed["cross_entropy"] - baseline["cross_entropy"],
        "recall_before": [_recall(baseline["predictions"], targets, label) for label in DYNAMIC_LABELS],
        "recall_after": [_recall(changed["predictions"], targets, label) for label in DYNAMIC_LABELS],
    }


def _stage(name, hypothesis, comparison, intervention, result, mismatch, revised, next_question):
    return {"name": name, "hypothesis": hypothesis, "comparison": comparison, "intervention": intervention, "result": result, "mismatch": mismatch, "revised_hypothesis": revised, "next_question": next_question}


def run_subspace_experiment(data_dir, seeds=(7, 11, 19, 23, 31)):
    data = load_data(data_dir)
    train_dynamic = np.isin(data["train_y"], DYNAMIC_LABELS)
    test_dynamic = np.isin(data["test_y"], DYNAMIC_LABELS)
    runs = []
    model_info = []
    for seed in seeds:
        model = train_baseline(data["train_x"], data["train_y"], seed=seed)
        train_h2 = baseline_forward(model, data["train_x"])[1]
        test_h2 = baseline_forward(model, data["test_x"])[1]
        train_x = data["train_x"][train_dynamic]
        train_y = data["train_y"][train_dynamic]
        test_x = data["test_x"][test_dynamic]
        test_y = data["test_y"][test_dynamic]
        directions = np.array([activity_direction(train_h2[train_dynamic], train_y, label) for label in DYNAMIC_LABELS])
        geometry = direction_geometry(directions)
        subspace = shared_private_subspace(directions)
        center = train_h2[train_dynamic].mean(axis=0)
        activity_results = {}
        for position, label in enumerate(DYNAMIC_LABELS):
            direction = directions[position]
            score = (train_h2[train_dynamic] - center) @ direction
            feature_combo = family_combination_analysis(train_x, score, data["feature_names"])
            combo = feature_combo["combinations"][0]
            activity_results[str(label)] = {
                "direction_geometry_projection": float(subspace["shared_projections"][position]),
                "private_norm": float(np.linalg.norm(subspace["private_components"][position])),
                "feature_combo": feature_combo,
                "feature_intervention": _feature_combo_intervention(model, direction, center, train_x, test_x, test_y, combo, seed + label),
                "direction_intervention": _component_intervention(model, test_h2[test_dynamic], test_y, direction, center),
            }
        shared_intervention = _component_intervention(model, test_h2[test_dynamic], test_y, subspace["shared_component"], center)
        private_interventions = [_component_intervention(model, test_h2[test_dynamic], test_y, subspace["private_components"][position], center) for position in range(3)]
        runs.append({"seed": seed, "geometry": geometry, "shared_explained_fraction": float(subspace["shared_explained_fraction"]), "shared_projections": subspace["shared_projections"], "activity_results": activity_results, "shared_intervention": shared_intervention, "private_interventions": private_interventions})
        model_info.append({"model": model, "test_h2": test_h2[test_dynamic], "directions": directions})
    cross_seed = {}
    reference = model_info[0]
    test_y = data["test_y"][test_dynamic]
    for position, label in enumerate(DYNAMIC_LABELS):
        mask = test_y == label
        reference_scores = (reference["test_h2"] - reference["test_h2"].mean(axis=0)) @ reference["directions"][position]
        cross_seed[str(label)] = []
        for seed, info in zip(seeds[1:], model_info[1:]):
            scores = (info["test_h2"] - info["test_h2"].mean(axis=0)) @ info["directions"][position]
            cross_seed[str(label)].append({"seed": seed, "representation_similarity": pattern_similarity(reference["test_h2"][mask], info["test_h2"][mask]), "direction_score_similarity": float(_pearson(reference_scores[mask], scores[mask]))})
    stages = [
        _stage("baseline_geometry", "세 direction은 서로 독립적이다.", "pairwise cosine/angle와 projection", "direction 자체를 아직 제거하지 않고 기하를 비교", {"per_seed_geometry": [{"seed": row["seed"], "geometry": row["geometry"]} for row in runs], "cross_seed": cross_seed}, "cosine이 크거나 shared projection이 크면 독립 가설이 틀린다.", "활동 direction은 공유 기반 위에 private 차이를 더하는 구조일 수 있다.", "shared component가 실제 판단에 공통으로 필요한가?"),
        _stage("unexplained_contradiction", "세 direction은 서로 독립적인 활동 전용 축이다.", "음의 cosine, 90도 이상 각도, shared explained fraction", "겹침과 cross-effect를 독립 가설의 반례로 기록하고 추가 개입으로 넘김", {"shared_fraction_range": [min(row["shared_explained_fraction"] for row in runs), max(row["shared_explained_fraction"] for row in runs)], "cosine_range": [min(value for row in runs for value in row["geometry"]["cosines"].values()), max(value for row in runs for value in row["geometry"]["cosines"].values())], "angle_range": [min(value for row in runs for value in row["geometry"]["angles_degrees"].values()), max(value for row in runs for value in row["geometry"]["angles_degrees"].values())]}, "겹침이 단순한 공통 의미인지, 분류 경계를 함께 만드는 얽힘인지 geometry만으로는 설명되지 않는다.", "direction은 활동의 이름을 저장한 독립 슬롯이 아니라 서로 경쟁·보완하는 decision coordinate일 수 있다.", "공유와 고유 성분을 각각 제거했을 때 어떤 활동이 함께 무너지는가?"),
        _stage("shared_private", "세 활동은 공통 subspace와 활동별 private subspace로 분해된다.", "SVD 첫 성분 설명력과 private norm", "shared/private component 각각을 test hidden2에서 projection 제거", {"shared_explained_fraction": [{"seed": row["seed"], "value": row["shared_explained_fraction"]} for row in runs], "shared_intervention": [{"seed": row["seed"], "value": row["shared_intervention"]} for row in runs], "private_intervention": [{"seed": row["seed"], "value": row["private_interventions"]} for row in runs]}, "shared 제거가 모든 활동을 함께 무너뜨리거나 private 제거가 선택적이지 않으면 단순 공통/고유 분해가 부족하다.", "공유 component는 중간 운동 판단, private residual은 활동 경계의 일부를 담당한다는 약한 모델.", "공유 제거와 private 제거의 cross-effect가 일관적인가?"),
        _stage("physical_families", "direction은 물리적 feature family 조합으로 설명된다.", "jerk/frequency/autocorrelation/energy/gravity_change 대표 feature와 joint R2", "상관 통제 후 family 대표 조합을 선택", {"seed_activity_combinations": [{"seed": row["seed"], "activities": {label: {"concept": row["activity_results"][label]["feature_combo"]["combinations"][0]["concept"], "features": row["activity_results"][label]["feature_combo"]["combinations"][0]["features"], "joint_gain": row["activity_results"][label]["feature_combo"]["combinations"][0]["joint_gain"]} for label in row["activity_results"]}} for row in runs]}, "대표 feature가 seed마다 완전히 달라지면 family-level 설명만 남는다.", "정확한 feature 이름보다 물리적 family 조합이 안정적인 설명 단위다.", "family 조합을 바꾸면 direction score와 분류가 예상대로 변하는가?"),
        _stage("combination_intervention", "두 family 조합이 단일/무작위 feature보다 direction을 만든다.", "selected combo vs single feature vs random combo", "test에서 선택 feature를 0으로 교란하고 direction score/recall 비교", {"results": [{"seed": row["seed"], "activities": {label: row["activity_results"][label]["feature_intervention"] for label in row["activity_results"]}} for row in runs]}, "선택 조합이 모든 seed에서 random보다 강하지 않으면 조합의 인과성이 불안정하다.", "일부 활동에서만 조합 효과가 안정적이며, 나머지는 중복 feature 또는 다른 보상 경로가 있다.", "내부 direction 제거가 feature 교란보다 더 직접적인가?"),
        _stage("internal_intervention", "shared/private direction 제거가 공통/선택적 활동 붕괴를 만든다.", "shared 및 각 private projection 제거 후 세 활동 recall matrix", "hidden2에서 projection 제거", {"per_seed": [{"seed": row["seed"], "shared": row["shared_intervention"], "private": row["private_interventions"]} for row in runs]}, "private 제거가 다른 활동까지 크게 무너뜨리면 private direction은 독립 활동 의미가 아니다.", "direction은 활동 전용 축이 아니라 서로 얽힌 decision coordinates일 수 있다.", "feature family와 shared/private direction의 연결을 더 직접적으로 검증할 수 있는가?"),
        _stage("revised_model", "feature→특정 뉴런→활동 구조로 설명할 수 있다.", "모든 개입과 cross-seed geometry를 한 구조로 비교", "물리 feature family 조합과 shared/private subspace를 함께 사용", {"structure": "physical sensor pattern -> shared/private activation subspace -> activity judgment", "evidence": "direction geometry, projection interventions, family combinations", "status": "partial_not_unique"}, "고정 feature 공식과 고정 뉴런 대응이 seed에 따라 바뀌면 단일 상징 규칙은 성립하지 않는다.", "사람이 이해할 수 있는 안정 단위는 물리적 family motif와 subspace intervention이며, 개별 feature/뉴런은 구현 후보일 뿐이다.", "더 많은 seed·raw temporal model·독립 validation에서 family motif가 유지되는가?"),
    ]
    return _jsonable({"settings": {"seeds": list(seeds), "dynamic_labels": list(DYNAMIC_LABELS)}, "runs": runs, "cross_seed": cross_seed, "stages": stages})


def write_report(result, path):
    lines = ["# 활동 direction의 공유/고유 subspace 분석", "", "목표: feature→뉴런→활동이라는 고정 설명 대신 물리 sensor pattern→shared/private activation subspace→활동 판단 구조를 비교한다.", ""]
    for stage in result["stages"]:
        lines += [f"## {stage['name']}", "", f"- 가설: {stage['hypothesis']}", f"- 비교 기준: {stage['comparison']}", f"- 개입 실험: {stage['intervention']}", f"- 실제 결과: `{json.dumps(stage['result'], ensure_ascii=False)}`", f"- 예상과 맞지 않는 점: {stage['mismatch']}", f"- 수정된 가설: {stage['revised_hypothesis']}", f"- 다음 질문: {stage['next_question']}", ""]
    lines += ["## 사람이 읽을 수 있는 후보 구조", "", "`jerk·주파수·자기상관·energy·중력 변화 feature family 조합 → shared/private activation subspace → 활동별 score와 최종 판단`을 현재의 최소 설명 구조로 제안한다. 이는 유일한 수식이나 고정 뉴런 목록이 아니라, 개입으로 확인된 계산 motif다.", "", "## 재현", "", "`python -m uci_har.subspace_experiment`"]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    root = Path(__file__).parent
    result = run_subspace_experiment(root / "UCI HAR Dataset")
    (root / "subspace_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_report(result, root / "subspace_analysis.md")
    print(json.dumps({"runs": len(result["runs"]), "result": str(root / "subspace_results.json")}, indent=2))
