"""Feature-combination and activation-direction interventions for UCI HAR."""

import json
from itertools import combinations
from pathlib import Path

import numpy as np

from .distributed_experiment import DYNAMIC_LABELS, _jsonable, _pearson, pattern_similarity
from .uci_har_experiment import CLASSES, baseline_forward, load_data, softmax, train_baseline


def activity_direction(activations, targets, label):
    own = activations[targets == label].mean(axis=0)
    other = activations[targets != label].mean(axis=0)
    direction = own - other
    norm = np.linalg.norm(direction)
    return direction / norm if norm else direction


def fit_r2(features, target):
    features = np.asarray(features, dtype=float)
    target = np.asarray(target, dtype=float)
    if features.ndim == 1:
        features = features[:, None]
    if np.var(target) == 0:
        return 0.0
    design = np.column_stack([np.ones(len(features)), features])
    prediction = design @ np.linalg.lstsq(design, target, rcond=None)[0]
    return float(1.0 - np.sum((target - prediction) ** 2) / np.sum((target - target.mean()) ** 2))


def correlation_control_selection(features, target, names, max_features=6, correlation_limit=0.85):
    scores = [abs(_pearson(features[:, index], target)) for index in range(features.shape[1])]
    order = np.argsort(scores)[::-1]
    selected = []
    for index in order:
        if all(abs(_pearson(features[:, index], features[:, row["index"]])) < correlation_limit for row in selected):
            selected.append({"index": int(index), "name": names[index], "direction_correlation": float(scores[index])})
        if len(selected) == max_features:
            break
    return selected


def direction_feature_analysis(features, target, names, max_features=6):
    selected = correlation_control_selection(features, target, names, max_features=max_features)
    pairs = []
    for first, second in combinations(selected, 2):
        single = max(fit_r2(features[:, [first["index"]]], target), fit_r2(features[:, [second["index"]]], target))
        joint = fit_r2(features[:, [first["index"], second["index"]]], target)
        pairs.append({"features": [first["name"], second["name"]], "indices": [first["index"], second["index"]], "single_r2": single, "joint_r2": joint, "joint_gain": joint - single})
    pairs.sort(key=lambda row: row["joint_gain"], reverse=True)
    return {"controlled_features": selected, "combination_r2": fit_r2(features[:, [row["index"] for row in selected]], target), "interactions": pairs[:10]}


def remove_direction_projection(values, direction, strength=1.0, center=None):
    values = np.asarray(values, dtype=float)
    direction = np.asarray(direction, dtype=float)
    center = np.zeros(values.shape[1]) if center is None else np.asarray(center, dtype=float)
    centered = values - center
    projection = (centered @ direction)[:, None] * direction
    return values - strength * projection


def _hidden2_outputs(model, hidden2, targets):
    probabilities = softmax(hidden2 @ model["w3"] + model["b3"])
    predictions = probabilities.argmax(axis=1)
    chosen = probabilities[np.arange(len(targets)), targets]
    return {
        "accuracy": float(np.mean(predictions == targets)),
        "cross_entropy": float(-np.log(np.clip(chosen, 1e-12, 1.0)).mean()),
        "predictions": predictions,
    }


def _recall(predictions, targets, label):
    mask = targets == label
    return float(np.mean(predictions[mask] == label)) if mask.any() else 0.0


def _feature_intervention(model, test_x, test_y, indices, seed):
    baseline = evaluate_outputs_from_model(model, test_x, test_y)
    selected = test_x.copy()
    selected[:, indices] = 0.0
    selected_metrics = evaluate_outputs_from_model(model, selected, test_y)
    single = test_x.copy()
    single[:, indices[:1]] = 0.0
    single_metrics = evaluate_outputs_from_model(model, single, test_y)
    rng = np.random.default_rng(seed)
    random_indices = rng.choice(test_x.shape[1], size=len(indices), replace=False)
    random = test_x.copy()
    random[:, random_indices] = 0.0
    random_metrics = evaluate_outputs_from_model(model, random, test_y)
    return {
        "selected": {"indices": indices, "accuracy": selected_metrics["accuracy"], "loss_change": selected_metrics["cross_entropy"] - baseline["cross_entropy"], "recall": [_recall(selected_metrics["predictions"], test_y, label) for label in DYNAMIC_LABELS]},
        "single": {"indices": indices[:1], "accuracy": single_metrics["accuracy"], "loss_change": single_metrics["cross_entropy"] - baseline["cross_entropy"]},
        "random": {"indices": [int(index) for index in random_indices], "accuracy": random_metrics["accuracy"], "loss_change": random_metrics["cross_entropy"] - baseline["cross_entropy"]},
        "baseline": {"accuracy": baseline["accuracy"], "recall": [_recall(baseline["predictions"], test_y, label) for label in DYNAMIC_LABELS]},
    }


def evaluate_outputs_from_model(model, inputs, targets):
    _, _, probabilities = baseline_forward(model, inputs)
    predictions = probabilities.argmax(axis=1)
    chosen = probabilities[np.arange(len(targets)), targets]
    return {"accuracy": float(np.mean(predictions == targets)), "cross_entropy": float(-np.log(np.clip(chosen, 1e-12, 1.0)).mean()), "predictions": predictions}


def _direction_intervention(model, hidden2, targets, direction, center, label, seed):
    baseline = _hidden2_outputs(model, hidden2, targets)
    selected_hidden = remove_direction_projection(hidden2, direction, center=center)
    selected = _hidden2_outputs(model, selected_hidden, targets)
    half_hidden = remove_direction_projection(hidden2, direction, strength=0.5, center=center)
    half = _hidden2_outputs(model, half_hidden, targets)
    rng = np.random.default_rng(seed)
    random_direction = rng.normal(size=direction.shape)
    random_direction /= np.linalg.norm(random_direction)
    random = _hidden2_outputs(model, remove_direction_projection(hidden2, random_direction, center=center), targets)
    return {
        "target_label": label,
        "baseline": {"accuracy": baseline["accuracy"], "target_recall": _recall(baseline["predictions"], targets, label)},
        "selected_full": {"accuracy": selected["accuracy"], "loss_change": selected["cross_entropy"] - baseline["cross_entropy"], "target_recall": _recall(selected["predictions"], targets, label)},
        "selected_half": {"accuracy": half["accuracy"], "loss_change": half["cross_entropy"] - baseline["cross_entropy"], "target_recall": _recall(half["predictions"], targets, label)},
        "random_direction": {"accuracy": random["accuracy"], "loss_change": random["cross_entropy"] - baseline["cross_entropy"], "target_recall": _recall(random["predictions"], targets, label)},
    }


def run_direction_experiment(data_dir, seeds=(7, 11, 19, 23, 31)):
    data = load_data(data_dir)
    train_mask = np.isin(data["train_y"], DYNAMIC_LABELS)
    test_mask = np.isin(data["test_y"], DYNAMIC_LABELS)
    runs = []
    model_info = []
    for seed in seeds:
        model = train_baseline(data["train_x"], data["train_y"], seed=seed)
        train_h1, train_h2, _ = baseline_forward(model, data["train_x"])
        test_h1, test_h2, _ = baseline_forward(model, data["test_x"])
        train_dynamic_x, train_dynamic_y = data["train_x"][train_mask], data["train_y"][train_mask]
        test_dynamic_x, test_dynamic_y = data["test_x"][test_mask], data["test_y"][test_mask]
        activity_results = {}
        directions = {}
        for label in DYNAMIC_LABELS:
            direction = activity_direction(train_h2[train_mask], train_dynamic_y, label)
            directions[label] = direction
            train_scores = (train_h2[train_mask] - train_h2[train_mask].mean(axis=0)) @ direction
            feature_analysis = direction_feature_analysis(train_dynamic_x, train_scores, data["feature_names"])
            selected_indices = [row["index"] for row in feature_analysis["controlled_features"][:4]]
            activity_results[str(label)] = {
                "direction_norm": float(np.linalg.norm(direction)),
                "train_score_mean": float(train_scores.mean()),
                "feature_analysis": feature_analysis,
                "feature_intervention": _feature_intervention(model, test_dynamic_x, test_dynamic_y, selected_indices, seed + label),
                "activation_intervention": _direction_intervention(model, test_h2[test_mask], test_dynamic_y, direction, train_h2[train_mask].mean(axis=0), label, seed + label + 100),
            }
        runs.append({"seed": seed, "activity_results": activity_results})
        model_info.append({"model": model, "test_h2": test_h2[test_mask], "directions": directions})
    similarity = {}
    reference = model_info[0]
    for label in DYNAMIC_LABELS:
        label_mask = data["test_y"][test_mask] == label
        similarity[str(label)] = []
        reference_direction = reference["directions"][label]
        reference_score = (reference["test_h2"] - reference["test_h2"].mean(axis=0)) @ reference_direction
        for seed, info in zip(seeds[1:], model_info[1:]):
            direction = info["directions"][label]
            score = (info["test_h2"] - info["test_h2"].mean(axis=0)) @ direction
            similarity[str(label)].append({"seed": seed, "representation_similarity": pattern_similarity(reference["test_h2"][label_mask], info["test_h2"][label_mask]), "direction_score_similarity": _pearson(reference_score[label_mask], score[label_mask])})
    stages = [
        {"name": "seed_stability", "hypothesis": "representation structure and activity directions repeat across five seeds even when neuron IDs differ.", "experiment": "pairwise sample distance geometry and direction-score correlations on identical test samples", "result": similarity, "mismatch": "low direction score similarity would weaken the common-direction claim.", "revised_hypothesis": "common geometry may be more stable than any named neuron or direction coordinate."},
        {"name": "activity_directions", "hypothesis": "each dynamic activity has a distinct distributed direction.", "experiment": "one-vs-rest hidden-layer-2 centroid contrast for each activity", "result": [{"seed": run["seed"], "activities": {label: {"top_features": [item["name"] for item in run["activity_results"][label]["feature_analysis"]["controlled_features"]], "combination_r2": run["activity_results"][label]["feature_analysis"]["combination_r2"]} for label in run["activity_results"]}} for run in runs], "mismatch": "direction concentration or seed-specific feature sets can contradict a single shared explanation.", "revised_hypothesis": "activity directions may be geometrically repeatable but feature implementations may vary."},
        {"name": "feature_combinations", "hypothesis": "feature combinations explain direction scores better than single features.", "experiment": "controlled feature selection and pair joint R2 gain", "result": [{"seed": run["seed"], "activities": {label: {"r2": run["activity_results"][label]["feature_analysis"]["combination_r2"], "interactions": run["activity_results"][label]["feature_analysis"]["interactions"][:3]} for label in run["activity_results"]}} for run in runs], "mismatch": "small joint gains mean the direction is mostly a single-feature or correlated proxy.", "revised_hypothesis": "some activity directions require a group, but the group may be redundant."},
        {"name": "correlation_control", "hypothesis": "the same feature groups remain after correlated duplicates are removed.", "experiment": "greedy correlation-controlled representatives with limit 0.85", "result": [{"seed": run["seed"], "activities": {label: [item["name"] for item in run["activity_results"][label]["feature_analysis"]["controlled_features"]] for label in run["activity_results"]}} for run in runs], "mismatch": "different representatives from the same correlated family are not independent evidence.", "revised_hypothesis": "feature families, rather than exact feature names, are the stable explanatory units."},
        {"name": "feature_intervention", "hypothesis": "joint feature perturbation selectively damages the matching activity more than a single or random feature.", "experiment": "test dynamic subset: controlled combination vs first feature vs random combination", "result": [{"seed": run["seed"], "activities": {label: run["activity_results"][label]["feature_intervention"] for label in run["activity_results"]}} for run in runs], "mismatch": "negative or seed-dependent changes weaken causal interpretation.", "revised_hypothesis": "feature combinations are necessary under this model, but not necessarily uniquely causal."},
        {"name": "activation_direction_intervention", "hypothesis": "removing one activity direction selectively reduces that activity recall more than removing a random direction.", "experiment": "full and half projection removal from hidden layer 2 on dynamic test subset", "result": [{"seed": run["seed"], "activities": {label: run["activity_results"][label]["activation_intervention"] for label in run["activity_results"]}} for run in runs], "mismatch": "broad accuracy loss without selective target recall loss means the direction is not activity-specific.", "revised_hypothesis": "directions are useful decision coordinates but may be entangled with neighboring activity boundaries."},
        {"name": "human_structure", "hypothesis": "temporal feature family combination→distributed direction score→activity judgment is a readable approximation.", "experiment": "report controlled feature representatives, joint R2, projection intervention, and target recall per activity", "result": {"structure": "feature family representatives -> one-vs-rest direction score -> six-class output", "caveat": "not a causal symbolic rule and not guaranteed unique across seeds"}, "mismatch": "seed variation and correlated feature families prevent a unique final formula.", "revised_hypothesis": "the stable object is a family-level computational motif, not a single formula."},
    ]
    return _jsonable({"settings": {"seeds": list(seeds), "dynamic_labels": list(DYNAMIC_LABELS)}, "runs": runs, "seed_similarity": similarity, "stages": stages})


def write_report(result, path):
    lines = ["# 활동별 분산 활성화 방향 분석", "", "가설: 여러 시간적 sensor feature의 조합이 분산 activation direction을 만들고, 그 방향이 세부 활동 판단을 형성한다.", ""]
    for stage in result["stages"]:
        lines += [f"## {stage['name']}", "", f"- 가설: {stage['hypothesis']}", f"- 실험: {stage['experiment']}", f"- 결과: `{json.dumps(stage['result'], ensure_ascii=False)}`", f"- 예상과 맞지 않는 점: {stage['mismatch']}", f"- 수정된 가설: {stage['revised_hypothesis']}", ""]
    lines += ["## 최종 구조", "", "`시간적 sensor feature family 조합 → one-vs-rest 분산 activation direction → 활동 판단`을 현재 모델을 설명하는 후보 구조로 기록한다. 다만 상관 feature의 대표 선택과 두 seed 이상의 재현성이 필요하며, feature family가 인과적 원인이라고 단정하지 않는다.", "", "## 재현", "", "`python -m uci_har.direction_experiment`"]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    root = Path(__file__).parent
    output = run_direction_experiment(root / "UCI HAR Dataset")
    (root / "direction_results.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    write_report(output, root / "direction_analysis.md")
    print(json.dumps({"runs": len(output["runs"]), "result": str(root / "direction_results.json")}, indent=2))
