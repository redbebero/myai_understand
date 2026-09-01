"""Human-readable temporal and relational model for raw UCI HAR signals."""

import json
from pathlib import Path

import numpy as np

from .raw_cnn_experiment import CLASSES, CHANNELS, softmax

WINDOWS, WINDOW_STEPS = 8, 16
CHANNEL_ROLES = ("mean", "std", "rms", "change", "trend", "stillness")
PAIR_ROLES = ("coactivation", "change_coactivation", "lagged_coactivation")
PAIRS = ((0, 3), (1, 4), (2, 5), (0, 6), (1, 7), (2, 8), (0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5))
EVENT_ROLES = ("energy", "change", "gyro_variability", "accel_gyro_coactivation", "direction_change", "stillness")
RAW_FEATURE_COUNT = WINDOWS * (CHANNELS * len(CHANNEL_ROLES) + len(PAIRS) * len(PAIR_ROLES))
STRUCTURED_FEATURE_COUNT = RAW_FEATURE_COUNT + WINDOWS * len(EVENT_ROLES)


def _channel_roles(signal):
    changes = np.diff(signal)
    return (signal.mean(), signal.std(), np.sqrt(np.mean(signal * signal)), np.mean(np.abs(changes)), signal[-1] - signal[0], np.mean(np.abs(changes) < 0.1))


def _pair_roles(left, right):
    left_change, right_change = np.diff(left), np.diff(right)
    lagged = left[:-1] * right[1:]
    return (np.mean(left * right), np.mean(left_change * right_change), np.mean(lagged))


def _event_roles(window):
    accel, gyro = window[:3], window[3:6]
    changes = np.diff(window, axis=1)
    return (np.sqrt(np.mean(accel * accel)), np.mean(np.abs(changes)), gyro.std(), np.mean(accel * gyro), np.mean(np.abs(np.diff(accel, axis=1))), np.mean(np.abs(changes) < 0.1))


def structured_role_names():
    names = []
    for window in range(WINDOWS):
        names.extend(f"window_{window}_channel_{channel}_{role}" for channel in range(CHANNELS) for role in CHANNEL_ROLES)
        names.extend(f"window_{window}_sensor_{left}_{right}_{role}" for left, right in PAIRS for role in PAIR_ROLES)
        names.extend(f"window_{window}_{role}" for role in EVENT_ROLES)
    return names


def structured_model_feature_names():
    names = structured_role_names()
    event_names = names[-WINDOWS * len(EVENT_ROLES):]
    return names + [f"above_threshold_{name}" for name in event_names]


def structured_role_features(inputs):
    values = np.asarray(inputs)
    features = []
    for window_index in range(WINDOWS):
        window = values[:, window_index * WINDOW_STEPS:(window_index + 1) * WINDOW_STEPS]
        for channel in window:
            features.extend(_channel_roles(channel))
        for left, right in PAIRS:
            features.extend(_pair_roles(window[left], window[right]))
        features.extend(_event_roles(window))
    return np.asarray(features)


def structured_role_matrix(inputs):
    return np.vstack([structured_role_features(row) for row in inputs])


def train_structured_model(x, y, epochs=300, learning_rate=0.04, seed=7):
    features = structured_role_matrix(x)
    mean, scale = features.mean(axis=0), features.std(axis=0)
    scale[scale == 0] = 1.0
    features = (features - mean) / scale
    event_start = WINDOWS * (CHANNELS * len(CHANNEL_ROLES) + len(PAIRS) * len(PAIR_ROLES))
    event_values = features[:, event_start:]
    thresholds = np.percentile(event_values, 75, axis=0)
    activated = np.maximum(event_values - thresholds, 0.0)
    features = np.column_stack((features, activated))
    rng = np.random.default_rng(seed)
    weights = rng.normal(0, 0.03, (features.shape[1], CLASSES))
    bias = np.zeros(CLASSES)
    for _ in range(epochs):
        probabilities = softmax(features @ weights + bias)
        error = probabilities.copy()
        error[np.arange(len(y)), y] -= 1.0
        error /= len(y)
        weights -= learning_rate * features.T @ error
        bias -= learning_rate * error.sum(axis=0)
    return {"weights": weights, "bias": bias, "mean": mean, "scale": scale, "thresholds": thresholds}


def _model_features(model, inputs):
    features = (structured_role_matrix(inputs) - model["mean"]) / model["scale"]
    event_start = RAW_FEATURE_COUNT
    activated = np.maximum(features[:, event_start:] - model["thresholds"], 0.0)
    return np.column_stack((features, activated))


def structured_predict(model, inputs):
    return np.argmax(_model_features(model, inputs) @ model["weights"] + model["bias"], axis=1)


def structured_accuracy(model, inputs, targets):
    return float(np.mean(structured_predict(model, inputs) == targets))


def model_parameter_count(model):
    return int(model["weights"].size + model["bias"].size)


def save_model(path, model):
    np.savez(path, **model)


if __name__ == "__main__":
    from .raw_cnn_experiment import load_raw

    root = Path(__file__).parent
    data = load_raw(root.parent / "data" / "UCI HAR Dataset")
    model = train_structured_model(data["train_x"], data["train_y"])
    print(json.dumps({"train": structured_accuracy(model, data["train_x"], data["train_y"]), "test": structured_accuracy(model, data["test_x"], data["test_y"]), "parameters": model_parameter_count(model), "features": STRUCTURED_FEATURE_COUNT}, indent=2))
