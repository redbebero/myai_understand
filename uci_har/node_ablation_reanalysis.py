"""Independent validation-selected hidden-node ablation reanalysis."""

import json
import math
from pathlib import Path

import numpy as np

from .uci_har_experiment import (
    CLASSES,
    HIDDEN,
    LABELS,
    ablate_hidden2,
    accuracy,
    baseline_forward,
    load_data,
    train_baseline,
)

SEEDS = (7, 11, 19, 23, 31)
VAL_SUBJECTS = (1, 3, 5, 6, 7, 8)


def paired_metrics(before, after, labels):
    before_correct = before == labels
    after_correct = after == labels
    improve = (~before_correct) & after_correct
    worsen = before_correct & (~after_correct)
    return {
        "accuracy_before": float(before_correct.mean()),
        "accuracy_after": float(after_correct.mean()),
        "difference": float(after_correct.mean() - before_correct.mean()),
        "improve_count": int(improve.sum()),
        "worsen_count": int(worsen.sum()),
        "unchanged_prediction_count": int(np.sum(before == after)),
    }


def bootstrap_ci(values, seed=123, draws=20000):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    samples = np.array([rng.choice(values, len(values), replace=True).mean() for _ in range(draws)])
    return [float(x) for x in np.quantile(samples, [0.025, 0.975])]


def exact_mcnemar_p(improve, worsen):
    tail = sum(math.comb(n, k) for k in range(min(improve, worsen) + 1)) / (2 ** n)
    return float(min(1.0, 2 * tail))


def node_drop_ranking(model, inputs, labels):
    baseline = accuracy(model, inputs, labels)
    rows = []
    for node in range(HIDDEN[1]):
        score = accuracy(ablate_hidden2(model, node), inputs, labels)
        rows.append({"node": node, "accuracy": score, "drop": baseline - score})
    return sorted(rows, key=lambda row: row["drop"])


def contribution_summary(model, inputs, labels, nodes):
    _, hidden, probabilities = baseline_forward(model, inputs)
    logits = np.log(np.clip(probabilities, 1e-12, 1.0))
    correct = logits.argmax(axis=1) == labels
    result = {}
    for node in nodes:
        contribution = hidden[:, node, None] * model["w3"][node][None, :]
        result[str(node)] = {
            "active_rate": float(np.mean(hidden[:, node] > 0)),
            "mean_activation": float(hidden[:, node].mean()),
            "mean_contribution_correct": contribution[correct].mean(axis=0).tolist(),
            "mean_contribution_wrong": contribution[~correct].mean(axis=0).tolist(),
        }
    return result


def run(root):
    root = Path(root)
    data = load_data(root / "UCI HAR Dataset")
    fit_mask = ~np.isin(data["subject_train"], VAL_SUBJECTS)
    val_mask = ~fit_mask
    runs = []
    for seed in SEEDS:
        model = train_baseline(data["train_x"][fit_mask], data["train_y"][fit_mask], seed=seed)
        ranking = node_drop_ranking(model, data["train_x"][val_mask], data["train_y"][val_mask])
        selected = [row["node"] for row in ranking[:2]]
        ablated = ablate_hidden2(ablate_hidden2(model, selected[0]), selected[1])
        base_test = np.asarray(baseline_forward(model, data["test_x"])[2]).argmax(axis=1)
        pair_test = np.asarray(baseline_forward(ablated, data["test_x"])[2]).argmax(axis=1)
        base_val = np.asarray(baseline_forward(model, data["train_x"][val_mask])[2]).argmax(axis=1)
        pair_val = np.asarray(baseline_forward(ablated, data["train_x"][val_mask])[2]).argmax(axis=1)
        runs.append({
            "seed": seed,
            "selected_nodes": selected,
            "validation": paired_metrics(base_val, pair_val, data["train_y"][val_mask]),
            "test": paired_metrics(base_test, pair_test, data["test_y"]),
            "test_contributions": contribution_summary(model, data["test_x"], data["test_y"], selected),
            "validation_ranking": ranking,
        })
    test_diffs = []
    for run_result in runs:
        test_diffs.append(run_result["test"]["difference"])
    result = {
        "design": {
            "seeds": list(SEEDS),
            "validation_subjects": list(VAL_SUBJECTS),
            "test_subjects": sorted(set(data["subject_test"].tolist())),
            "selection": "lowest two individual hidden2 ablation drops on validation subjects",
            "test_used_for_selection": False,
        },
        "runs": runs,
        "summary": {
            "mean_test_difference": float(np.mean(test_diffs)),
            "median_test_difference": float(np.median(test_diffs)),
            "seedwise_test_differences": test_diffs,
            "bootstrap_ci_95": bootstrap_ci(test_diffs),
            "seeds_with_positive_test_difference": int(np.sum(np.asarray(test_diffs) > 0)),
        },
    }
    (root / "node_ablation_reanalysis_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    output = run(Path(__file__).parent)
    print(json.dumps(output["summary"], indent=2))
