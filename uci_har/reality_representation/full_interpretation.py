"""Generate a complete human-readable interpretation of all latent mappings."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from symbolic_aggregation import parse_expression, readable

HERE = Path(__file__).resolve().parent
INPUT = HERE / "results" / "bottleneck_gate" / "latent_symbolic_results.json"
OUTPUT = HERE / "FULL_INTERPRETATION_REPORT.md"


def region_text(ref):
    seconds = (ref["end"] - ref["start"]) / 50.0
    return f"`{ref['channel']}[{ref['start']}:{ref['end']}]` ({seconds:.2f} s window)"


def main():
    data = json.loads(INPUT.read_text())
    rows = []
    recurrence = Counter()
    for run in data["runs"]:
        for latent in run["latent_results"]:
            expressions = []
            for item in latent["expressions"]["features"][:3]:
                operation, refs = parse_expression(item["name"])
                meaning = readable(operation, refs)
                locations = " and ".join(region_text(ref) for ref in refs)
                expressions.append({"formula": item["name"], "meaning": meaning, "locations": locations, "coefficient": item["coefficient"], "r2": latent["expressions"]["validation_r2"]})
                recurrence[(operation, tuple(ref["channel"] for ref in refs))] += 1
            rows.append({"variant": run["variant"], "seed": run["seed"], "dimension": latent["dimension"], "r2": latent["expressions"]["validation_r2"], "expressions": expressions})
    lines = ["# Complete Human Interpretation of AI-Discovered Latent Values", "", "This report translates every latent mapping's top three automatically selected expressions. The expressions were generated from raw channel-time regions; the human-readable wording is applied only after selection.", "", "## Overall result", "", f"- bottleneck dimension: **k={data['bottleneck_k']}**", f"- latent subspace similarity proxy: mean **{data['latent_subspace_proxy_mean']:.3f}**, minimum **{data['latent_subspace_proxy_min']:.3f}**", f"- latent-to-formula validation R²: mean **{sum(row['r2'] for row in rows)/len(rows):.3f}**, range **{min(row['r2'] for row in rows):.3f}–{max(row['r2'] for row in rows):.3f}**", "", "## Recurring raw structures", "", "| operation + channels | recurrence among 40 latent mappings | automatic interpretation |", "|---|---:|---|"]
    for (operation, channels), count in recurrence.most_common(20):
        if operation == "local_squared_magnitude": meaning = "local signal magnitude"
        elif operation == "local_temporal_change": meaning = "local temporal change"
        elif operation == "local_range": meaning = "local movement range"
        elif operation == "local_mean_level": meaning = "local mean level"
        elif operation == "cross_signal_coordination": meaning = "simultaneous change between signals"
        elif operation == "joint_signal_strength": meaning = "joint signal strength"
        else: meaning = operation
        lines.append(f"| `{operation}` on {', '.join(channels)} | {count}/40 | {meaning} |")
    lines += ["", "## Every latent mapping", ""]
    for row in rows:
        lines += [f"### {row['variant']} seed {row['seed']} — latent dimension z{row['dimension']+1}", "", f"Validation R² for this latent: **{row['r2']:.3f}**", ""]
        for index, item in enumerate(row["expressions"], 1):
            lines += [f"{index}. Formula: `{item['formula']}`", f"   - Raw location: {item['locations']}", f"   - Automatic structural interpretation: **{item['meaning']}**", f"   - Lasso coefficient: `{item['coefficient']:.4f}`", ""]
    lines += ["## What the person can conclude", "", "The AI repeatedly found short raw channel-time regions and simple local operations that predict parts of the learned four-dimensional representation. The most common human-readable interpretations are local signal magnitude, local range, local temporal change, and local mean level. These are interpretations of the discovered formulas, not feature names supplied during training.", "", "## Boundary", "", "A latent dimension is not a universal physical variable: latent coordinates can rotate between models, and the subspace similarity is moderate. The report therefore treats recurring formula structures and raw regions as candidates for measurement definitions, not as proven causes of the activity distinction.", ""]
    OUTPUT.write_text("\n".join(lines))
    print(f"wrote {OUTPUT} with {len(rows)} latent mappings")


if __name__ == "__main__":
    main()
