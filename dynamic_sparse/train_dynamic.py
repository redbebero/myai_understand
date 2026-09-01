import copy
import json
import math
import random
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def tanh_derivative(value):
    return 1 - value * value


def sigmoid(value):
    return 1 / (1 + math.exp(-max(-60, min(60, value))))


def matrix(rows, columns, rng):
    return [[rng.uniform(-0.8, 0.8) for _ in range(columns)] for _ in range(rows)]


def new_model(seed=7, width=8, density=0.5):
    rng = random.Random(seed)
    shapes = [(width, 2), (width, width), (1, width)]
    weights = [matrix(rows, columns, rng) for rows, columns in shapes]
    masks = []
    for values in weights:
        positions = [(r, c) for r in range(len(values)) for c in range(len(values[0]))]
        rng.shuffle(positions)
        active = round(len(positions) * density)
        mask = [[0.0 for _ in row] for row in values]
        for r, c in positions[:active]:
            mask[r][c] = 1.0
        masks.append(mask)
    return {
        "architecture": [2, width, width, 1],
        "weights": weights,
        "masks": masks,
        "biases": [[0.0] * width, [0.0] * width, [0.0]],
        "rewire_events": [],
    }


def forward(model, inputs):
    values = inputs
    hidden_values = []
    for layer in range(2):
        values = [
            math.tanh(sum(model["weights"][layer][r][c] * model["masks"][layer][r][c] * values[c]
                          for c in range(len(values))) + model["biases"][layer][r])
            for r in range(len(model["weights"][layer]))
        ]
        hidden_values.append(values)
    output = sigmoid(sum(model["weights"][2][0][c] * model["masks"][2][0][c] * values[c]
                         for c in range(len(values))) + model["biases"][2][0])
    return hidden_values[0], hidden_values[1], output


def evaluate(model, rows):
    return sum((forward(model, row["inputs"])[2] >= 0.5) == bool(row["target"])
               for row in rows) / len(rows)


def gradients(model, inputs, target):
    h1, h2, output = forward(model, inputs)
    error = output - target
    gradient = [
        [[0.0 for _ in row] for row in layer]
        for layer in model["weights"]
    ]
    gradient[2][0] = [error * value for value in h2]
    error2 = [error * model["weights"][2][0][i] * model["masks"][2][0][i] * tanh_derivative(h2[i])
              for i in range(len(h2))]
    for r in range(len(h2)):
        for c in range(len(h1)):
            gradient[1][r][c] = error2[r] * h1[c]
    error1 = [sum(model["weights"][1][r][c] * model["masks"][1][r][c] * error2[r]
                  for r in range(len(error2))) * tanh_derivative(h1[c])
              for c in range(len(h1))]
    for r in range(len(h1)):
        for c in range(len(inputs)):
            gradient[0][r][c] = error1[r] * inputs[c]
    return gradient, error1, error2, error


def train(rows, seed=7, epochs=900, learning_rate=0.04, rewire_every=100, rewires=1, width=8, density=0.5):
    model = new_model(seed, width=width, density=density)
    rng = random.Random(seed + 1)
    gradient_scores = [[[0.0 for _ in row] for row in layer] for layer in model["weights"]]
    history = []
    for epoch in range(1, epochs + 1):
        order = list(range(len(rows)))
        rng.shuffle(order)
        for index in order:
            row = rows[index]
            gradient, error1, error2, error = gradients(model, row["inputs"], row["target"])
            for layer in range(3):
                for r in range(len(gradient[layer])):
                    for c in range(len(gradient[layer][r])):
                        value = gradient[layer][r][c]
                        gradient_scores[layer][r][c] += abs(value)
                        if model["masks"][layer][r][c]:
                            model["weights"][layer][r][c] -= learning_rate * value
            model["biases"][2][0] -= learning_rate * error
            for r in range(len(model["biases"][1])):
                model["biases"][1][r] -= learning_rate * error2[r]
            for r in range(len(model["biases"][0])):
                model["biases"][0][r] -= learning_rate * error1[r]
        if epoch % rewire_every == 0:
            changes = []
            for layer in range(3):
                active = [(abs(model["weights"][layer][r][c]), r, c)
                          for r in range(len(model["weights"][layer]))
                          for c in range(len(model["weights"][layer][r]))
                          if model["masks"][layer][r][c]]
                inactive = [(gradient_scores[layer][r][c], r, c)
                            for r in range(len(model["weights"][layer]))
                            for c in range(len(model["weights"][layer][r]))
                            if not model["masks"][layer][r][c]]
                for _, r, c in sorted(active)[:rewires]:
                    model["masks"][layer][r][c] = 0.0
                    model["weights"][layer][r][c] = 0.0
                    changes.append(f"remove L{layer}[{r},{c}]")
                for _, r, c in sorted(inactive, reverse=True)[:rewires]:
                    model["masks"][layer][r][c] = 1.0
                    model["weights"][layer][r][c] = rng.uniform(-0.05, 0.05)
                    changes.append(f"grow L{layer}[{r},{c}]")
                for r in range(len(gradient_scores[layer])):
                    for c in range(len(gradient_scores[layer][r])):
                        gradient_scores[layer][r][c] = 0.0
            model["rewire_events"].append({"epoch": epoch, "changes": changes})
            history.append({"epoch": epoch, "train_accuracy": evaluate(model, rows)})
    return model, history


def active_connections(model):
    return sum(sum(row) for layer in model["masks"] for row in layer)


if __name__ == "__main__":
    root = Path(__file__).parent
    train_rows = load(root / "train.json")
    test_rows = load(root / "test.json")
    model, history = train(train_rows, width=12)
    result = {
        "architecture": model["architecture"],
        "active_connections": active_connections(model),
        "train_accuracy": evaluate(model, train_rows),
        "test_accuracy": evaluate(model, test_rows),
        "rewire_events": model["rewire_events"],
        "history": history,
    }
    (root / "model.json").write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    (root / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("architecture", "active_connections", "train_accuracy", "test_accuracy")}, indent=2))
