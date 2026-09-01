"""Turn learned directions into signed, class-conditional time patterns."""
import json
from pathlib import Path

import numpy as np

from experiment import LABELS, ROOT, VAL_SUBJECTS, accuracy, forward, load_raw, standardize, task_directions, train

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"


def channel_curve(x, channel):
    return x[:, channel, :]
def describe_pair(x, y, channel, feature_name, label_a, label_b):
    a = channel_curve(x[y == label_a], channel).mean(0)
    b = channel_curve(x[y == label_b], channel).mean(0)
    delta = a - b
    return {"channel": channel, "feature_name": feature_name,
            "label_a": LABELS[label_a], "label_b": LABELS[label_b],
            "mean_delta_a_minus_b": float(delta.mean()),
            "peak_abs_delta": float(np.max(np.abs(delta))),
            "peak_time_index": int(np.argmax(np.abs(delta))),
            "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "trajectory_a": a.tolist(), "trajectory_b": b.tolist(),
            "trajectory_delta": delta.tolist()}


def run(seed=7):
    train_x, train_y, train_subject, test_x, test_y, test_subject = load_raw()
    train_x, test_x = standardize(train_x, test_x)
    val = np.isin(train_subject, VAL_SUBJECTS)
    fit = ~val
    model = train(train_x[fit], train_y[fit], seed, variant="flat")
    effects = []
    hidden = forward(model, test_x)[1]
    dirs = task_directions(model, hidden.shape[1])
    logits = hidden @ model["w3"] + model["b3"]
    for axis in range(dirs.shape[1]):
        component = np.outer(hidden @ dirs[:, axis], dirs[:, axis])
        without = (hidden - component) @ model["w3"] + model["b3"]
        margin = np.mean(np.abs((logits[:, 0] - logits[:, 1]) - (without[:, 0] - without[:, 1])))
        effects.append(float(margin))
    chosen_direction = int(np.argmax(effects))
    pairs = [(0, 1), (3, 4), (1, 2)]
    channels = {"total_acc_x": 6, "total_acc_y": 7, "total_acc_z": 8,
                "body_acc_x": 0, "body_acc_y": 1, "body_acc_z": 2,
                "body_gyro_y": 4, "body_gyro_z": 5}
    analyses = []
    for label_a, label_b in pairs:
        selected = ["total_acc_x", "body_acc_x", "body_acc_y"] if (label_a, label_b) == (0, 1) else ["total_acc_y", "body_acc_y", "total_acc_z"]
        for name in selected:
            analyses.append(describe_pair(test_x, test_y, channels[name], name, label_a, label_b))
    result = {"seed": seed, "test_accuracy": accuracy(model, test_x, test_y),
              "chosen_direction_for_walking_pair": chosen_direction,
              "direction_pair_margin_effects": effects,
              "pair_patterns": analyses,
              "protocol_note": "signed class-conditional test trajectories; descriptive, not causal"}
    OUT.mkdir(exist_ok=True)
    (OUT / "temporal_pattern_results.json").write_text(json.dumps(result, indent=2))
    return result


def build_report(result):
    lines = ["# Signed Temporal Pattern Analysis", "",
             f"- seed: {result['seed']}",
             f"- test accuracy: {result['test_accuracy']:.4f}",
             f"- selected direction for walking pair: {result['chosen_direction_for_walking_pair']}", ""]
    for item in result["pair_patterns"]:
        direction = "higher" if item["mean_delta_a_minus_b"] > 0 else "lower"
        lines.append(f"- {item['label_a']} vs {item['label_b']}: {item['feature_name']} mean trajectory is {abs(item['mean_delta_a_minus_b']):.4f} {direction} for the first activity; largest absolute difference at timestep {item['peak_time_index']} (magnitude {item['peak_abs_delta']:.4f}).")
    lines += ["", "These are class-conditional descriptive patterns. They do not establish that the named channel causes the classifier decision.", ""]
    (HERE / "TEMPORAL_REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    build_report(run())
