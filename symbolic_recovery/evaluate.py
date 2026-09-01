import json
from pathlib import Path

from recover import model_output
from simple_formula import predict


ROOT = Path(__file__).parent


def evaluate(path):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    model_correct = sum((model_output(row["inputs"]) >= 0.5) == bool(row["target"]) for row in rows)
    formula_correct = sum((predict(row["inputs"]) >= 0.5) == bool(row["target"]) for row in rows)
    return {
        "examples": len(rows),
        "model_accuracy": model_correct / len(rows),
        "formula_accuracy": formula_correct / len(rows),
        "model_formula_agreement": sum(
            (model_output(row["inputs"]) >= 0.5) == (predict(row["inputs"]) >= 0.5)
            for row in rows
        ) / len(rows),
    }


if __name__ == "__main__":
    result = evaluate(ROOT.parent / "two_spiral" / "spiral_test.json")
    (ROOT / "labeled_evaluation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
