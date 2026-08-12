import json
import math
import random
import copy
from pathlib import Path


def load_dataset(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sigmoid(value):
    value = max(-60.0, min(60.0, value))
    return 1 / (1 + math.exp(-value))


def forward(model, inputs):
    hidden1 = [
        math.tanh(sum(w * x for w, x in zip(row, inputs)) + bias)
        for row, bias in zip(model["weights_input_hidden1"], model["bias_hidden1"])
    ]
    hidden2 = [
        math.tanh(sum(w * x for w, x in zip(row, hidden1)) + bias)
        for row, bias in zip(model["weights_hidden1_hidden2"], model["bias_hidden2"])
    ]
    output = sigmoid(sum(w * x for w, x in zip(model["weights_hidden2_output"], hidden2)) + model["bias_output"])
    return hidden1, hidden2, output


def predict(model, inputs):
    return forward(model, inputs)[2]


def evaluate(model, rows):
    correct = sum((predict(model, row["inputs"]) >= 0.5) == bool(row["target"]) for row in rows)
    return correct / len(rows)


def new_model(seed=7, width=12):
    rng = random.Random(seed)
    scale = 0.8
    return {
        "architecture": [2, width, width, 1],
        "weights_input_hidden1": [[rng.uniform(-scale, scale) for _ in range(2)] for _ in range(width)],
        "bias_hidden1": [0.0] * width,
        "weights_hidden1_hidden2": [[rng.uniform(-scale, scale) for _ in range(width)] for _ in range(width)],
        "bias_hidden2": [0.0] * width,
        "weights_hidden2_output": [rng.uniform(-scale, scale) for _ in range(width)],
        "bias_output": 0.0,
    }


def train(rows, epochs=1200, learning_rate=0.03, seed=7, width=12, initial_model=None):
    model = copy.deepcopy(initial_model) if initial_model is not None else new_model(seed, width)
    rng = random.Random(seed + 1)
    for _ in range(epochs):
        order = list(range(len(rows)))
        rng.shuffle(order)
        for index in order:
            inputs, target = rows[index]["inputs"], rows[index]["target"]
            hidden1, hidden2, output = forward(model, inputs)
            output_error = output - target
            old_output_weights = model["weights_hidden2_output"][:]
            for j, value in enumerate(hidden2):
                model["weights_hidden2_output"][j] -= learning_rate * output_error * value
            model["bias_output"] -= learning_rate * output_error

            error2 = [output_error * weight * (1 - value * value) for weight, value in zip(old_output_weights, hidden2)]
            old_layer2 = [row[:] for row in model["weights_hidden1_hidden2"]]
            for j, error in enumerate(error2):
                for i, value in enumerate(hidden1):
                    model["weights_hidden1_hidden2"][j][i] -= learning_rate * error * value
                model["bias_hidden2"][j] -= learning_rate * error

            error1 = []
            for i, value in enumerate(hidden1):
                backprop = sum(old_layer2[j][i] * error2[j] for j in range(len(error2)))
                error1.append(backprop * (1 - value * value))
            for i, error in enumerate(error1):
                for j, value in enumerate(inputs):
                    model["weights_input_hidden1"][i][j] -= learning_rate * error * value
                model["bias_hidden1"][i] -= learning_rate * error
    return model


def prune_connection(model, layer, row, column):
    keys = ["weights_input_hidden1", "weights_hidden1_hidden2", "weights_hidden2_output"]
    if layer == 2:
        model[keys[layer]][column] = 0.0
    else:
        model[keys[layer]][row][column] = 0.0


def prune_neuron(model, layer, index):
    if layer == 1:
        model["weights_input_hidden1"][index] = [0.0] * len(model["weights_input_hidden1"][index])
        model["weights_hidden1_hidden2"][index] = [0.0] * len(model["weights_hidden1_hidden2"][index])
    elif layer == 2:
        model["weights_hidden1_hidden2"][index] = [0.0] * len(model["weights_hidden1_hidden2"][index])
        model["weights_hidden2_output"][index] = 0.0
    else:
        raise ValueError("layer must be 1 or 2")


def nonzero_weights(model):
    return sum(
        value != 0.0
        for key in ("weights_input_hidden1", "weights_hidden1_hidden2", "weights_hidden2_output")
        for row in (model[key] if isinstance(model[key][0], list) else [model[key]])
        for value in row
    )


if __name__ == "__main__":
    root = Path(__file__).parent
    rows = load_dataset(root / "spiral_train.json")
    model = train(rows)
    (root / "spiral_model.json").write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote spiral_model.json; train accuracy={evaluate(model, rows):.1%}")
