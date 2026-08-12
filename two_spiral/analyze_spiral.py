import copy
import json
from pathlib import Path

from generate_spirals import generate
from train_spiral import evaluate, forward, load_dataset, new_model, nonzero_weights, prune_connection, prune_neuron, train


ROOT = Path(__file__).parent


def run():
    train_path = ROOT / "spiral_train.json"
    test_path = ROOT / "spiral_test.json"
    if not train_path.exists() or not test_path.exists():
        generate(train_path, seed=7)
        generate(test_path, seed=19)
    train_rows = load_dataset(train_path)
    test_rows = load_dataset(test_path)
    model = train(train_rows)
    (ROOT / "spiral_model.json").write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    baseline_train = evaluate(model, train_rows)
    baseline_test = evaluate(model, test_rows)

    samples = []
    for row in test_rows[:4]:
        hidden1, hidden2, output = forward(model, row["inputs"])
        samples.append({"inputs": row["inputs"], "target": row["target"], "hidden1": hidden1, "hidden2": hidden2, "output": output})

    connection_results = []
    for layer, rows, columns, label in ((0, 12, 2, "input→hidden1"), (1, 12, 12, "hidden1→hidden2"), (2, 1, 12, "hidden2→output")):
        for row in range(rows):
            for column in range(columns):
                candidate = copy.deepcopy(model)
                prune_connection(candidate, layer, row, column)
                connection_results.append({"connection": f"{label}[{row},{column}]", "test_accuracy": evaluate(candidate, test_rows)})

    neuron_results = []
    for layer in (1, 2):
        for index in range(12):
            candidate = copy.deepcopy(model)
            prune_neuron(candidate, layer, index)
            neuron_results.append({"neuron": f"hidden{layer}[{index}]", "test_accuracy": evaluate(candidate, test_rows)})

    least_damaging_connection = max(connection_results, key=lambda item: item["test_accuracy"])
    most_damaging_neuron = min(neuron_results, key=lambda item: item["test_accuracy"])
    candidate = copy.deepcopy(model)
    prune_neuron(candidate, 1, int(most_damaging_neuron["neuron"].split("[")[1][:-1]))
    retrained = train(train_rows, seed=7, width=12, initial_model=candidate)
    pruned_retrained_test = evaluate(retrained, test_rows)

    repeats = []
    for seed in (11, 23, 41):
        repeated = train(train_rows, epochs=300, seed=seed, width=12)
        repeats.append({"seed": seed, "train_accuracy": evaluate(repeated, train_rows), "test_accuracy": evaluate(repeated, test_rows)})

    analysis = {
        "baseline": {"train_accuracy": baseline_train, "test_accuracy": baseline_test, "nonzero_weights": nonzero_weights(model)},
        "samples": samples,
        "connection_results": connection_results,
        "neuron_results": neuron_results,
        "least_damaging_single_connection_removal": least_damaging_connection,
        "most_damaging_single_neuron_removal": most_damaging_neuron,
        "retrained_reference_test_accuracy": pruned_retrained_test,
        "repeat_training_results": repeats,
    }
    (ROOT / "spiral_analysis.json").write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Two-Spiral Model Analysis",
        "",
        "## Question",
        "",
        "Can a model learn a pattern that is difficult to express with simple rules, and can its essential computation be identified by removing parts of the model?",
        "",
        "## Baseline",
        "",
        "Architecture: 2 inputs → 12 hidden neurons → 12 hidden neurons → 1 output",
        f"Training examples: {len(train_rows)}",
        f"Test examples: {len(test_rows)}",
        f"Training accuracy: {baseline_train:.1%}",
        f"Test accuracy: {baseline_test:.1%}",
        f"Nonzero weights: {nonzero_weights(model)}",
        "",
        "## Computation samples",
        "",
        "The full intermediate activations are saved in `spiral_analysis.json`. They show how each input becomes hidden-layer values and then an output probability.",
        "",
        "## Removal results",
        "",
        f"Least damaging single connection removal: `{least_damaging_connection['connection']}`; accuracy {least_damaging_connection['test_accuracy']:.1%}.",
        f"Most damaging single neuron removal: `{most_damaging_neuron['neuron']}`; accuracy {most_damaging_neuron['test_accuracy']:.1%}.",
        f"Test accuracy after pruning that neuron and retraining: {pruned_retrained_test:.1%}.",
        "",
        "Repeated training results are saved in `spiral_analysis.json` for seeds 11, 23, and 41.",
        "",
        "## Interpretation",
        "",
        "The two-spiral task is larger than XOR and includes unseen test points. A component that can be removed with little accuracy loss may be redundant for this trained model. A component that causes a large loss contributes to the current computation, but is not automatically necessary in every possible model.",
        "",
        "The next valid comparison is pruning followed by retraining. The model may recover performance by redistributing the calculation across other neurons. This distinguishes a currently used connection from a structurally essential connection.",
        "",
        "## Limitations",
        "",
        "This experiment uses one architecture, one training seed, and one data-generation process. Repeat with other seeds and noise levels before claiming a general rule.",
    ]
    (ROOT / "spiral_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run()
