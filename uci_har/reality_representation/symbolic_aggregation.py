"""Translate and aggregate automatically discovered raw expressions."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

CHANNELS = ("body_acc_x", "body_acc_y", "body_acc_z", "body_gyro_x", "body_gyro_y", "body_gyro_z", "total_acc_x", "total_acc_y", "total_acc_z")
HERE = Path(__file__).resolve().parent
INPUT = HERE / "results" / "advanced_trace" / "auto_structure_results.json"
OUTPUT = HERE / "results" / "advanced_trace" / "symbolic_aggregation_results.json"
REPORT = HERE / "SYMBOLIC_AGGREGATION_REPORT.md"
REF = re.compile(r"x\[(\d+),(\d+):(\d+)\]")


def parse_expression(name):
    seen = set()
    refs = []
    for c, a, b in REF.findall(name):
        key = (int(c), int(a), int(b))
        if key in seen:
            continue
        seen.add(key)
        refs.append({"channel_index": key[0], "channel": CHANNELS[key[0]], "start": key[1], "end": key[2]})
    if name.startswith("mean(square("):
        operation = "local_squared_magnitude"
    elif name.startswith("mean(abs(diff("):
        operation = "local_temporal_change"
    elif name.startswith("max("):
        operation = "local_range"
    elif name.startswith("slope("):
        operation = "local_trend"
    elif name.startswith("corr("):
        operation = "cross_signal_coordination"
    elif "*x[" in name:
        operation = "joint_signal_strength"
    elif name.startswith("mean("):
        operation = "local_mean_level"
    else:
        operation = "other"
    return operation, refs


def overlap(a, b):
    left, right = max(a["start"], b["start"]), min(a["end"], b["end"])
    union = max(a["end"], b["end"]) - min(a["start"], b["start"])
    return max(0, right - left) / max(union, 1)


def same_region(a, b):
    if len(a) != len(b):
        return False
    return all(x["channel_index"] == y["channel_index"] and overlap(x, y) >= 0.25 for x, y in zip(a, b))


def readable(operation, refs):
    if operation == "local_squared_magnitude":
        return f"{refs[0]['channel']} local signal magnitude"
    if operation == "local_temporal_change":
        return f"{refs[0]['channel']} local temporal change"
    if operation == "local_range":
        return f"{refs[0]['channel']} local movement range"
    if operation == "local_trend":
        return f"{refs[0]['channel']} local increasing/decreasing trend"
    if operation == "cross_signal_coordination":
        return f"{refs[0]['channel']} and {refs[1]['channel']} simultaneous change"
    if operation == "joint_signal_strength":
        return f"joint signal strength of {refs[0]['channel']} and {refs[1]['channel']}"
    return f"{refs[0]['channel']} local mean level"


def run():
    data = json.loads(INPUT.read_text())
    clusters = []
    for run in data["runs"]:
        # Use the five largest Lasso coefficients per checkpoint; dense tails are not treated as repeated concepts.
        for feature in run["expressions"]["features"][:5]:
            operation, refs = parse_expression(feature["name"])
            if not refs:
                continue
            # Pair references are sorted by channel index for stable matching.
            refs = sorted(refs, key=lambda ref: (ref["channel_index"], ref["start"]))
            found = None
            for cluster in clusters:
                if cluster["operation"] == operation and same_region(cluster["refs"], refs):
                    found = cluster
                    break
            if found is None:
                found = {"operation": operation, "refs": refs, "members": []}
                clusters.append(found)
            found["members"].append({"variant": run["variant"], "seed": run["seed"], "expression": feature["name"], "coefficient": feature["coefficient"], "validation_r2": run["expressions"]["validation_r2"]})
    for cluster in clusters:
        cluster["meaning"] = readable(cluster["operation"], cluster["refs"])
        cluster["model_count"] = len({(m["variant"], m["seed"]) for m in cluster["members"]})
        cluster["mean_abs_coefficient"] = sum(abs(m["coefficient"]) for m in cluster["members"]) / len(cluster["members"])
        cluster["mean_coefficient"] = sum(m["coefficient"] for m in cluster["members"]) / len(cluster["members"])
    clusters.sort(key=lambda c: (-c["model_count"], -c["mean_abs_coefficient"]))
    result = {"source": str(INPUT), "aggregation": "same operation, same raw channels, overlapping time windows", "clusters": clusters}
    OUTPUT.write_text(json.dumps(result, indent=2))
    lines = ["# Symbolic Explanation Aggregation", "", "Automatically discovered expressions were parsed, mapped to raw channel names and time windows, then clustered only when operation, channels, and windows matched approximately.", "", "| readable concept | raw structure | model recurrence | mean coefficient |", "|---|---|---:|---:|"]
    for c in clusters[:12]:
        refs_text = ", ".join(f"{r['channel']}[{r['start']}:{r['end']}]" for r in c["refs"])
        lines.append(f"| {c['meaning']} | `{c['operation']}` on {refs_text} | {c['model_count']}/10 | {c['mean_coefficient']:.3f} |")
    lines += ["", "## Interpretation boundary", "", "The readable meaning is generated from the algebraic operator and raw channel identity. It is a label for the discovered computation, not a claim about biological causality. Region clustering uses only raw coordinate overlap and does not impose the earlier human-defined feature families.", ""]
    REPORT.write_text("\n".join(lines))
    return result


if __name__ == "__main__":
    result = run()
    print(len(result["clusters"]), "clusters")
