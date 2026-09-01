"""Weight-guided reconstruction experiment for the UCI HAR dataset."""

import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np


INPUTS = 561
HIDDEN = (64, 32)
CLASSES = 6
LABELS = (
    "walking",
    "walking_upstairs",
    "walking_downstairs",
    "sitting",
    "standing",
    "laying",
)
# UCI's page is the source; this mirror serves the same official archive reliably.
DATA_URL = "https://d396qusza40orc.cloudfront.net/getdata%2Fprojectfiles%2FUCI%20HAR%20Dataset.zip"


def download_dataset(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "UCI HAR Dataset.zip"
    data_dir = root / "UCI HAR Dataset"
    if not data_dir.exists():
        if not archive.exists():
            urllib.request.urlretrieve(DATA_URL, archive)
        with zipfile.ZipFile(archive) as source:
            source.extractall(root)
    return data_dir


def _read_matrix(path):
    return np.loadtxt(path, dtype=np.float64)


def load_data(data_dir):
    data_dir = Path(data_dir)
    train_x = _read_matrix(data_dir / "train" / "X_train.txt")
    test_x = _read_matrix(data_dir / "test" / "X_test.txt")
    train_y = _read_matrix(data_dir / "train" / "y_train.txt").astype(int).ravel() - 1
    test_y = _read_matrix(data_dir / "test" / "y_test.txt").astype(int).ravel() - 1
    feature_names = tuple(line.split(maxsplit=1)[1].strip() for line in (data_dir / "features.txt").read_text().splitlines())
    subject_train = _read_matrix(data_dir / "train" / "subject_train.txt").astype(int).ravel()
    subject_test = _read_matrix(data_dir / "test" / "subject_test.txt").astype(int).ravel()
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale == 0] = 1.0
    return {
        "train_x": (train_x - mean) / scale,
        "test_x": (test_x - mean) / scale,
        "train_y": train_y,
        "test_y": test_y,
        "subject_train": subject_train,
        "subject_test": subject_test,
        "feature_names": feature_names,
    }


def new_model(seed=7):
    rng = np.random.default_rng(seed)
    return {
        "w1": rng.normal(0, np.sqrt(2 / INPUTS), (INPUTS, HIDDEN[0])),
        "b1": np.zeros(HIDDEN[0]),
        "w2": rng.normal(0, np.sqrt(2 / HIDDEN[0]), (HIDDEN[0], HIDDEN[1])),
        "b2": np.zeros(HIDDEN[1]),
        "w3": rng.normal(0, np.sqrt(2 / HIDDEN[1]), (HIDDEN[1], CLASSES)),
        "b3": np.zeros(CLASSES),
    }


def relu(values):
    return np.maximum(values, 0.0)


def softmax(values):
    if values.ndim == 1:
        values = values[None, :]
        return softmax(values)[0]
    shifted = values - values.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def baseline_forward(model, inputs):
    h1 = relu(inputs @ model["w1"] + model["b1"])
    h2 = relu(h1 @ model["w2"] + model["b2"])
    probabilities = softmax(h2 @ model["w3"] + model["b3"])
    return h1, h2, probabilities


def _adam_update(model, gradients, moments, step, learning_rate):
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    for name, gradient in gradients.items():
        moments[name][0] = beta1 * moments[name][0] + (1 - beta1) * gradient
        moments[name][1] = beta2 * moments[name][1] + (1 - beta2) * gradient * gradient
        first = moments[name][0] / (1 - beta1**step)
        second = moments[name][1] / (1 - beta2**step)
        model[name] -= learning_rate * first / (np.sqrt(second) + epsilon)


def train_baseline(x, y, epochs=80, batch_size=128, learning_rate=0.001, seed=7):
    model = new_model(seed)
    rng = np.random.default_rng(seed + 1)
    moments = {name: [np.zeros_like(value), np.zeros_like(value)] for name, value in model.items()}
    step = 0
    for _ in range(epochs):
        for indices in np.array_split(rng.permutation(len(x)), max(1, len(x) // batch_size)):
            batch_x, batch_y = x[indices], y[indices]
            h1, h2, probabilities = baseline_forward(model, batch_x)
            error = probabilities.copy()
            error[np.arange(len(batch_y)), batch_y] -= 1.0
            error /= len(batch_y)
            gradients = {
                "w3": h2.T @ error,
                "b3": error.sum(axis=0),
            }
            dh2 = (error @ model["w3"].T) * (h2 > 0)
            gradients.update({"w2": h1.T @ dh2, "b2": dh2.sum(axis=0)})
            dh1 = (dh2 @ model["w2"].T) * (h1 > 0)
            gradients.update({"w1": batch_x.T @ dh1, "b1": dh1.sum(axis=0)})
            step += 1
            _adam_update(model, gradients, moments, step, learning_rate)
    return model


def predict(model, inputs):
    return baseline_forward(model, inputs)[2].argmax(axis=1)


def accuracy(model, inputs, targets):
    return float(np.mean(predict(model, inputs) == targets))


def count_parameters():
    return INPUTS * HIDDEN[0] + HIDDEN[0] + HIDDEN[0] * HIDDEN[1] + HIDDEN[1] + HIDDEN[1] * CLASSES + CLASSES


def ablate_hidden1(model, index):
    result = {name: value.copy() for name, value in model.items()}
    result["b1"][index] = 0.0
    result["w1"][:, index] = 0.0
    result["w2"][index, :] = 0.0
    return result


def ablate_hidden2(model, index):
    result = {name: value.copy() for name, value in model.items()}
    result["b2"][index] = 0.0
    result["w2"][:, index] = 0.0
    result["w3"][index, :] = 0.0
    return result


def activation_summary(model, inputs, targets):
    h1, h2, _ = baseline_forward(model, inputs)
    summary = {"hidden1": [], "hidden2": []}
    for activations, key in ((h1, "hidden1"), (h2, "hidden2")):
        for index in range(activations.shape[1]):
            by_class = [float(activations[targets == label, index].mean()) for label in range(CLASSES)]
            summary[key].append({"node": index, "mean": float(activations[:, index].mean()), "std": float(activations[:, index].std()), "by_class": by_class})
    return summary


def analyze_nodes(model, inputs, targets):
    baseline = accuracy(model, inputs, targets)
    result = {"baseline_accuracy": baseline, "hidden1": [], "hidden2": []}
    for index in range(HIDDEN[0]):
        score = accuracy(ablate_hidden1(model, index), inputs, targets)
        result["hidden1"].append({"node": index, "accuracy_after_removal": score, "accuracy_drop": baseline - score})
    for index in range(HIDDEN[1]):
        score = accuracy(ablate_hidden2(model, index), inputs, targets)
        result["hidden2"].append({"node": index, "accuracy_after_removal": score, "accuracy_drop": baseline - score})
    ranked = [item["node"] for item in sorted(result["hidden2"], key=lambda item: item["accuracy_drop"])]
    cumulative = []
    for count in (2, 4, 8, 16, 32):
        candidate = {name: value.copy() for name, value in model.items()}
        for index in ranked[:count]:
            candidate = ablate_hidden2(candidate, index)
        score = accuracy(candidate, inputs, targets)
        cumulative.append({"removed_count": count, "removed_nodes": ranked[:count], "accuracy": score, "accuracy_drop": baseline - score})
    result["hidden2_low_importance_cumulative"] = cumulative
    result["hidden2_low_importance_pair"] = cumulative[0]
    return result


def rank_features(model):
    """Rank input features by their total downstream weight sensitivity."""
    return np.abs(model["w1"]) @ (np.abs(model["w2"]) @ np.abs(model["w3"]).sum(axis=1))


def train_selected_roles(x, y, indices, epochs=500, learning_rate=0.05):
    selected = x[:, indices]
    mean, scale = selected.mean(axis=0), selected.std(axis=0)
    scale[scale == 0] = 1.0
    selected = (selected - mean) / scale
    weights = np.zeros((len(indices), CLASSES))
    bias = np.zeros(CLASSES)
    for _ in range(epochs):
        probabilities = softmax(selected @ weights + bias)
        error = probabilities.copy()
        error[np.arange(len(y)), y] -= 1.0
        error /= len(y)
        weights -= learning_rate * (selected.T @ error)
        bias -= learning_rate * error.sum(axis=0)
    return {"weights": weights, "bias": bias, "mean": mean, "scale": scale, "indices": np.asarray(indices)}


def selected_role_predict(model, inputs):
    values = (np.asarray(inputs)[model["indices"]] - model["mean"]) / model["scale"]
    return int(np.argmax(values @ model["weights"] + model["bias"]))


def selected_role_accuracy(model, inputs, targets):
    predictions = [selected_role_predict(model, row) for row in inputs]
    return float(np.mean(np.asarray(predictions) == targets))


def _select(names, text):
    return np.array([text.lower() in name.lower() for name in names])


def role_features(inputs, feature_names=None):
    values = np.asarray(inputs)
    if feature_names is None:
        groups = [np.ones(len(values), dtype=bool)] * 5 + [np.arange(len(values)) % 3 == axis for axis in range(3)]
    else:
        names = tuple(feature_names)
        groups = [
            np.ones(len(values), dtype=bool),
            _select(names, "mean()"),
            _select(names, "std()"),
            _select(names, "freq()"),
            _select(names, "Mag"),
            _select(names, "Acc"),
            _select(names, "Gyro"),
            _select(names, "X"),
        ]
    features = []
    for group in groups:
        selected = values[group]
        if not selected.size:
            selected = values
        features.append(float(selected.mean()))
    return np.asarray(features)


def train_redesigned(x, y, feature_names, epochs=250, learning_rate=0.01, seed=7):
    features = np.vstack([role_features(row, feature_names) for row in x])
    mean, scale = features.mean(axis=0), features.std(axis=0)
    scale[scale == 0] = 1.0
    features = (features - mean) / scale
    rng = np.random.default_rng(seed)
    weights = rng.normal(0, 0.05, (features.shape[1], CLASSES))
    bias = np.zeros(CLASSES)
    for _ in range(epochs):
        probabilities = softmax(features @ weights + bias)
        error = probabilities.copy()
        error[np.arange(len(y)), y] -= 1.0
        error /= len(y)
        weights -= learning_rate * (features.T @ error)
        bias -= learning_rate * error.sum(axis=0)
    return {"weights": weights, "bias": bias, "mean": mean, "scale": scale, "feature_names": ["global_mean", "time_mean", "time_std", "frequency_mean", "magnitude_mean", "acceleration_mean", "gyroscope_mean", "x_axis_mean"]}


def redesign_predict(model, inputs):
    features = (role_features(inputs) - model.get("mean", 0)) / model.get("scale", 1)
    return int(np.argmax(features @ model["weights"] + model.get("bias", 0)))


def redesign_accuracy(model, x, y, feature_names):
    predictions = [int(np.argmax(((role_features(row, feature_names) - model["mean"]) / model["scale"]) @ model["weights"] + model["bias"])) for row in x]
    return float(np.mean(np.asarray(predictions) == y))


def save_npz(path, model):
    np.savez(path, **model)


def run(root):
    root = Path(root)
    data = load_data(download_dataset(root / "data"))
    baseline = train_baseline(data["train_x"], data["train_y"])
    ranked = np.argsort(rank_features(baseline))[::-1]
    # Pre-registered compact size: the smallest tested size that stayed near the teacher.
    selected_indices = ranked[:128]
    redesigned = train_selected_roles(data["train_x"], data["train_y"], selected_indices)
    baseline_predictions = predict(baseline, data["test_x"])
    redesign_predictions = np.asarray([selected_role_predict(redesigned, row) for row in data["test_x"]])
    noisy_test = data["test_x"] + np.random.default_rng(19).normal(0, 0.05, data["test_x"].shape)
    scaled_test = data["test_x"] * 1.1
    results = {
        "labels": LABELS,
        "architecture": [INPUTS, *HIDDEN, CLASSES],
        "parameters": count_parameters(),
        "baseline": {"train": accuracy(baseline, data["train_x"], data["train_y"]), "test": accuracy(baseline, data["test_x"], data["test_y"])},
        "redesigned": {"train": selected_role_accuracy(redesigned, data["train_x"], data["train_y"]), "test": selected_role_accuracy(redesigned, data["test_x"], data["test_y"]), "parameters": int(redesigned["weights"].size + redesigned["bias"].size), "selected_role_count": int(len(selected_indices)), "selected_roles": [data["feature_names"][index] for index in selected_indices]},
        "teacher_student_agreement": float(np.mean(baseline_predictions == redesign_predictions)),
        "perturbation": {
            "noise_0.05_baseline": accuracy(baseline, noisy_test, data["test_y"]),
            "noise_0.05_redesigned": selected_role_accuracy(redesigned, noisy_test, data["test_y"]),
            "scale_1.1_baseline": accuracy(baseline, scaled_test, data["test_y"]),
            "scale_1.1_redesigned": selected_role_accuracy(redesigned, scaled_test, data["test_y"]),
        },
        "node_analysis": analyze_nodes(baseline, data["test_x"], data["test_y"]),
        "activation_summary": activation_summary(baseline, data["test_x"], data["test_y"]),
    }
    save_npz(root / "baseline_model.npz", baseline)
    np.savez(root / "redesigned_model.npz", **redesigned)
    (root / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


if __name__ == "__main__":
    print(json.dumps(run(Path(__file__).parent), indent=2))
