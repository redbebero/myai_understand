import json
import math
import random
from pathlib import Path


ROOT = Path(__file__).parent
MODEL = json.loads((ROOT / "target_model.json").read_text(encoding="utf-8"))
WEIGHTS = [
    MODEL["weights_input_hidden1"],
    MODEL["weights_hidden1_hidden2"],
    [MODEL["weights_hidden2_output"]],
]
BIASES = [
    MODEL["bias_hidden1"],
    MODEL["bias_hidden2"],
    [MODEL["bias_output"]],
]


def sigmoid(value):
    return 1 / (1 + math.exp(-max(-60.0, min(60.0, value))))


def model_output(inputs):
    values = list(inputs)
    for layer in range(2):
        values = [
            math.tanh(sum(
                WEIGHTS[layer][row][column] * values[column]
                for column in range(len(values))
            ) + BIASES[layer][row])
            for row in range(len(WEIGHTS[layer]))
        ]
    return sigmoid(sum(
        WEIGHTS[2][0][column] * values[column]
        for column in range(len(values))
    ) + BIASES[2][0])


def generate_probes(count, seed):
    rng = random.Random(seed)
    return [{"inputs": [rng.uniform(-1, 1), rng.uniform(-1, 1)], "output": None} for _ in range(count)]


def polar(inputs):
    x, y = inputs
    return math.hypot(x, y), math.atan2(y, x)


def formula_output(formula, inputs):
    radius, angle = polar(inputs)
    score = formula["polarity"] * math.cos(angle - formula["frequency"] * radius + formula["phase"])
    return sigmoid(formula["scale"] * score)


def formula_class(formula, inputs):
    return formula_output(formula, inputs) >= 0.5


def recover_formula(probes):
    observations = [(probe["inputs"], model_output(probe["inputs"])) for probe in probes]
    best = None
    for frequency_index in range(4, 81):
        frequency = frequency_index / 4
        for phase_index in range(32):
            phase = phase_index * 2 * math.pi / 32
            for polarity in (-1, 1):
                candidate = {
                    "type": "polar_cosine",
                    "frequency": frequency,
                    "phase": phase,
                    "polarity": polarity,
                    "scale": 8.0,
                    "complexity": 3,
                }
                matches = sum(
                    formula_class(candidate, inputs) == (output >= 0.5)
                    for inputs, output in observations
                )
                score = matches / len(observations)
                if best is None or score > best["agreement"]:
                    best = {**candidate, "agreement": score}
    return best


def agreement(formula, probes):
    return sum(
        formula_class(formula, probe["inputs"]) == (model_output(probe["inputs"]) >= 0.5)
        for probe in probes
    ) / len(probes)


def run():
    probes = generate_probes(1200, seed=7)
    for probe in probes:
        probe["output"] = model_output(probe["inputs"])
    train_probes, test_probes = probes[:600], probes[600:]
    formula = recover_formula(train_probes)
    result = {
        "information_used": ["target_model.json", "model architecture", "activation functions"],
        "labels_used_during_recovery": False,
        "probe_count": len(probes),
        "search_probe_count": len(train_probes),
        "verification_probe_count": len(test_probes),
        "formula": formula,
        "search_agreement": agreement(formula, train_probes),
        "verification_agreement": agreement(formula, test_probes),
        "model_formula": "sigmoid(8*cos(atan2(y,x)-frequency*hypot(x,y)+phase))",
    }
    (ROOT / "probe_outputs.json").write_text(json.dumps(probes, indent=2) + "\n", encoding="utf-8")
    (ROOT / "recovered_formula.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
