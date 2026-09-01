"""Build a concise report from the saved experiment JSON."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def build(source=HERE / "results" / "experiment_results.json", target=HERE / "REPORT.md"):
    data = json.loads(Path(source).read_text())
    lines = ["# Reality Representation Experiment", "", "## Protocol", ""]
    for key, value in data["protocol"].items():
        lines.append(f"- **{key}:** {value}")
    lines += ["", "## Results", "", "| variant | mean baseline | mean relation-broken | mean drop | mean intervention | mean control |", "|---|---:|---:|---:|---:|---:|"]
    for variant in data["protocol"]["variants"]:
        rows = [r for r in data["runs"] if r["variant"] == variant]
        baseline = sum(r["test_accuracy"] for r in rows) / len(rows)
        broken = sum(r["relation_broken_accuracy"] for r in rows) / len(rows)
        drop = sum(r["relation_drop"] for r in rows) / len(rows)
        intervention = sum(r["intervention"]["true_class_probability_change"] for r in rows) / len(rows)
        control = sum(r["matched_control"]["true_class_probability_change"] for r in rows) / len(rows)
        lines.append(f"| {variant} | {baseline:.4f} | {broken:.4f} | {drop:.4f} | {intervention:.4f} | {control:.4f} |")
    lines += ["", "## Recurring candidate features", ""]
    lines.extend(f"- {name}" for name in data["candidate_recurring_features"])
    lines += ["", "## Human-readable interpretation cards", ""]
    for row in data["runs"][:2]:
        lines.append(f"### {row['variant']} / seed {row['seed']}")
        for card in row["interpretation_cards"][:8]:
            strongest = max(card["influence_by_activity_pair"], key=card["influence_by_activity_pair"].get)
            lines.append(f"- Direction {card['latent_direction']}: **{card['human_abstraction']}**; strongest pair effect: `{strongest}`; required information: " +
                         ", ".join(item["name"] for item in card["required_sensor_information"]))
    lines += ["", "## Interpretation boundary", "", "Relation destruction supports dependence on temporal/channel structure. Human-readable cards describe candidate abstractions; causal claims require selective intervention beyond the current permutation control.", ""]
    Path(target).write_text("\n".join(lines))


if __name__ == "__main__":
    build()
