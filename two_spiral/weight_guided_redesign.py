"""Select important neurons from learned weights, freeze their equations, retrain the rest."""

import copy
import json
import math
import random
from pathlib import Path

from train_spiral import evaluate, forward, load_dataset, new_model, train


def importance(model, rows):
    baseline = evaluate(model, rows)
    scores = []
    for index in range(len(model["weights_input_hidden1"])):
        candidate = copy.deepcopy(model)
        candidate["weights_input_hidden1"][index] = [0.0, 0.0]
        candidate["weights_hidden1_hidden2"] = [
            [0.0 if column == index else value for column, value in enumerate(row)]
            for row in candidate["weights_hidden1_hidden2"]
        ]
        scores.append({"index": index, "accuracy_after_removal": evaluate(candidate, rows), "accuracy_drop": baseline - evaluate(candidate, rows)})
    return sorted(scores, key=lambda item: item["accuracy_drop"], reverse=True)


def fixed_forward(model, inputs, selected):
    input_indices = selected if max(selected, default=-1) < len(model["weights_input_hidden1"]) else range(len(model["weights_input_hidden1"]))
    hidden1 = [
        math.tanh(sum(w * x for w, x in zip(model["weights_input_hidden1"][index], inputs)) + model["bias_hidden1"][index])
        for index in input_indices
    ]
    hidden2 = [
        math.tanh(sum(w * x for w, x in zip(row, hidden1)) + bias)
        for row, bias in zip(model["weights_hidden1_hidden2"], model["bias_hidden2"])
    ]
    output = 1 / (1 + math.exp(-max(-60.0, min(60.0, sum(w * x for w, x in zip(model["weights_hidden2_output"], hidden2)) + model["bias_output"]))))
    return hidden1, hidden2, output


def fixed_evaluate(model, rows, selected):
    return sum((fixed_forward(model, row["inputs"], selected)[2] >= 0.5) == bool(row["target"]) for row in rows) / len(rows)


def retrain_downstream(source, rows, selected, epochs=1200, learning_rate=0.03, seed=17):
    rng = random.Random(seed)
    model = {
        "weights_input_hidden1": [source["weights_input_hidden1"][i][:] for i in selected],
        "bias_hidden1": [source["bias_hidden1"][i] for i in selected],
        "weights_hidden1_hidden2": [[rng.uniform(-0.8, 0.8) for _ in selected] for _ in source["bias_hidden2"]],
        "bias_hidden2": [0.0] * len(source["bias_hidden2"]),
        "weights_hidden2_output": [rng.uniform(-0.8, 0.8) for _ in source["bias_hidden2"]],
        "bias_output": 0.0,
    }
    order_rng = random.Random(seed + 1)
    for _ in range(epochs):
        order = list(range(len(rows)))
        order_rng.shuffle(order)
        for row_index in order:
            inputs, target = rows[row_index]["inputs"], rows[row_index]["target"]
            hidden1, hidden2, output = fixed_forward(model, inputs, selected)
            output_error = output - target
            old_output = model["weights_hidden2_output"][:]
            for j, value in enumerate(hidden2):
                model["weights_hidden2_output"][j] -= learning_rate * output_error * value
            model["bias_output"] -= learning_rate * output_error
            error2 = [output_error * weight * (1 - value * value) for weight, value in zip(old_output, hidden2)]
            for j, error in enumerate(error2):
                for i, value in enumerate(hidden1):
                    model["weights_hidden1_hidden2"][j][i] -= learning_rate * error * value
                model["bias_hidden2"][j] -= learning_rate * error
    return model


def readable_equations(model, selected):
    return [
        f"node_{position} = tanh(({model['weights_input_hidden1'][index][0]:.6g})*x + ({model['weights_input_hidden1'][index][1]:.6g})*y + ({model['bias_hidden1'][index]:.6g}))  # original hidden1[{index}]"
        for position, index in enumerate(selected)
    ]


def run(root):
    source = json.loads((root / "spiral_model.json").read_text(encoding="utf-8"))
    train_rows = load_dataset(root / "spiral_train.json")
    test_rows = load_dataset(root / "spiral_test.json")
    ranking = importance(source, train_rows)
    selected = [item["index"] for item in ranking[:6]]
    redesigned = retrain_downstream(source, train_rows, selected)
    size_sweep = []
    for count in (4, 6, 8, 10, 12):
        chosen = [item["index"] for item in ranking[:count]]
        candidate = retrain_downstream(source, train_rows, chosen, seed=17)
        size_sweep.append({
            "selected_count": count,
            "train_accuracy": fixed_evaluate(candidate, train_rows, chosen),
            "test_accuracy": fixed_evaluate(candidate, test_rows, chosen),
        })
    result = {
        "baseline": {"train_accuracy": evaluate(source, train_rows), "test_accuracy": evaluate(source, test_rows)},
        "importance_ranking": ranking,
        "selected_hidden1_nodes": selected,
        "size_sweep": size_sweep,
        "redesigned": {
            "architecture": [2, len(selected), 12, 1],
            "frozen_node_equations": readable_equations(source, selected),
            "trainable_part": "hidden1_to_hidden2 and hidden2_to_output only",
            "trainable_parameter_count": len(selected) * 12 + 12 + 12 + 1,
            "train_accuracy": fixed_evaluate(redesigned, train_rows, selected),
            "test_accuracy": fixed_evaluate(redesigned, test_rows, selected),
        },
    }
    (root / "weight_guided_redesign_model.json").write_text(json.dumps({"selected": selected, "model": redesigned}, indent=2) + "\n", encoding="utf-8")
    (root / "weight_guided_redesign_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(Path(__file__).parent), indent=2))
