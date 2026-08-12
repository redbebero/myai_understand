import json
import math
import random
from pathlib import Path


def sigmoid(value):
    value = max(-60.0, min(60.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def load_dataset(path="xor_dataset.json"):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def predict(model, inputs):
    hidden = [
        sigmoid(sum(weight * value for weight, value in zip(row, inputs)) + bias)
        for row, bias in zip(model["hidden_weights"], model["hidden_bias"])
    ]
    return sigmoid(sum(weight * value for weight, value in zip(model["output_weights"], hidden)) + model["output_bias"])


def evaluate(model, rows):
    return sum((predict(model, row["inputs"]) >= 0.5) == bool(row["target"])
               for row in rows) / len(rows)


def train(rows, epochs=10_000, seed=7):
    rng = random.Random(seed)
    model = {
        "hidden_weights": [[rng.uniform(-1, 1) for _ in range(2)] for _ in range(2)],
        "hidden_bias": [0.0, 0.0],
        "output_weights": [rng.uniform(-1, 1) for _ in range(2)],
        "output_bias": 0.0,
    }
    learning_rate = 1.0

    for _ in range(epochs):
        for row in rows:
            inputs, target = row["inputs"], row["target"]
            hidden = [
                sigmoid(sum(weight * value for weight, value in zip(weights, inputs)) + bias)
                for weights, bias in zip(model["hidden_weights"], model["hidden_bias"])
            ]
            output = sigmoid(sum(weight * value for weight, value in zip(model["output_weights"], hidden)) + model["output_bias"])
            output_error = (output - target) * output * (1 - output)

            old_output_weights = model["output_weights"][:]
            for index in range(2):
                model["output_weights"][index] -= learning_rate * output_error * hidden[index]
            model["output_bias"] -= learning_rate * output_error

            for hidden_index, hidden_value in enumerate(hidden):
                hidden_error = output_error * old_output_weights[hidden_index] * hidden_value * (1 - hidden_value)
                for input_index, value in enumerate(inputs):
                    model["hidden_weights"][hidden_index][input_index] -= learning_rate * hidden_error * value
                model["hidden_bias"][hidden_index] -= learning_rate * hidden_error
    return model


def prune_connection(model, layer, row, column):
    if layer == 0:
        model["hidden_weights"][row][column] = 0.0
    elif layer == 1:
        model["output_weights"][column] = 0.0
    else:
        raise ValueError("layer must be 0 or 1")


def nonzero_parameters(model):
    weights = model["hidden_weights"] + [model["output_weights"]]
    return sum(value != 0.0 for row in weights for value in row)


if __name__ == "__main__":
    dataset = load_dataset()
    model = train(dataset)
    Path("xor_model.json").write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote xor_model.json; accuracy={evaluate(model, dataset):.0%}")
