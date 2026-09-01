"""Raw-sensor Conv1D reconstruction experiment using NumPy only."""

import json
from pathlib import Path

import numpy as np


CHANNELS, STEPS, FILTERS, KERNEL, CLASSES = 9, 128, 12, 9, 6
LABELS = ("walking", "walking_upstairs", "walking_downstairs", "sitting", "standing", "laying")
SIGNALS = ("body_acc_x", "body_acc_y", "body_acc_z", "body_gyro_x", "body_gyro_y", "body_gyro_z", "total_acc_x", "total_acc_y", "total_acc_z")
WINDOWS, WINDOW_STEPS = 8, 16
WINDOW_ROLES = ("mean", "std", "rms", "mean_abs_change", "trend", "frequency_peak", "peak_count", "zero_crossings", "max_abs", "max_change", "still_fraction", "periodicity")
CROSS_WINDOW_ROLES = ("accel_rms", "gyro_rms", "total_acc_rms", "accel_gyro_alignment", "cross_channel_std", "simultaneous_change")
GLOBAL_ROLES = ("global_rms", "global_mean_abs_change", "global_still_fraction")


def load_raw(data_dir):
    data_dir = Path(data_dir)
    def load(split):
        folder = data_dir / split / "Inertial Signals"
        channels = [np.loadtxt(folder / f"{signal}_{split}.txt", dtype=np.float64) for signal in SIGNALS]
        return np.stack(channels, axis=1), np.loadtxt(data_dir / split / f"y_{split}.txt", dtype=int) - 1
    train_x, train_y = load("train")
    test_x, test_y = load("test")
    mean = train_x.mean(axis=(0, 2), keepdims=True)
    scale = train_x.std(axis=(0, 2), keepdims=True)
    scale[scale == 0] = 1.0
    return {"train_x": (train_x - mean) / scale, "test_x": (test_x - mean) / scale, "train_y": train_y, "test_y": test_y}


def new_model(seed=7):
    rng = np.random.default_rng(seed)
    return {
        "kernels": rng.normal(0, np.sqrt(2 / (CHANNELS * KERNEL)), (FILTERS, CHANNELS, KERNEL)),
        "bias": np.zeros(FILTERS),
        "output": rng.normal(0, np.sqrt(2 / FILTERS), (FILTERS, CLASSES)),
        "output_bias": np.zeros(CLASSES),
    }


def softmax(values):
    shifted = values - values.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def _windows(inputs):
    return np.lib.stride_tricks.sliding_window_view(inputs, KERNEL, axis=2)


def cnn_forward(model, inputs):
    windows = _windows(inputs)
    convolution = np.einsum("nclk,fck->nlf", windows, model["kernels"]).transpose(0, 2, 1) + model["bias"][None, :, None]
    activations = np.maximum(convolution, 0.0)
    pooled = activations.mean(axis=2)
    probabilities = softmax(pooled @ model["output"] + model["output_bias"])
    return activations, pooled, probabilities


def _adam_update(model, gradients, moments, step, learning_rate):
    for name, gradient in gradients.items():
        first, second = moments[name]
        first[:] = 0.9 * first + 0.1 * gradient
        second[:] = 0.999 * second + 0.001 * gradient * gradient
        model[name] -= learning_rate * (first / (1 - 0.9**step)) / (np.sqrt(second / (1 - 0.999**step)) + 1e-8)


def train_model(x, y, epochs=35, batch_size=128, learning_rate=0.003, seed=7):
    model = new_model(seed)
    rng = np.random.default_rng(seed + 1)
    moments = {name: (np.zeros_like(value), np.zeros_like(value)) for name, value in model.items()}
    step = 0
    for _ in range(epochs):
        for indices in np.array_split(rng.permutation(len(x)), max(1, len(x) // batch_size)):
            batch_x, batch_y = x[indices], y[indices]
            activations, pooled, probabilities = cnn_forward(model, batch_x)
            error = probabilities.copy()
            error[np.arange(len(batch_y)), batch_y] -= 1.0
            error /= len(batch_y)
            gradients = {"output": pooled.T @ error, "output_bias": error.sum(axis=0)}
            pooled_error = error @ model["output"].T / activations.shape[2]
            activation_error = pooled_error[:, :, None] * (activations > 0)
            gradients["kernels"] = np.einsum("nclk,nfl->fck", _windows(batch_x), activation_error)
            gradients["bias"] = activation_error.sum(axis=(0, 2))
            step += 1
            _adam_update(model, gradients, moments, step, learning_rate)
    return model


def predict(model, inputs):
    return cnn_forward(model, inputs)[2].argmax(axis=1)


def accuracy(model, inputs, targets):
    return float(np.mean(predict(model, inputs) == targets))


def count_parameters():
    return FILTERS * CHANNELS * KERNEL + FILTERS + FILTERS * CLASSES + CLASSES


def quantized_model(model, bits):
    limit = 2 ** (bits - 1) - 1
    result = {}
    for name, values in model.items():
        scale = np.max(np.abs(values)) / limit
        scale = scale if scale else 1.0
        result[name] = np.round(values / scale).clip(-limit, limit) * scale
    return result


def quantized_accuracy(model, inputs, targets):
    return accuracy(model, inputs, targets)


def ablate_filter(model, index):
    result = {name: value.copy() for name, value in model.items()}
    result["kernels"][index] = 0.0
    result["bias"][index] = 0.0
    result["output"][index] = 0.0
    return result


def ablate_filters(model, indices):
    result = {name: value.copy() for name, value in model.items()}
    for index in indices:
        result["kernels"][index] = 0.0
        result["bias"][index] = 0.0
        result["output"][index] = 0.0
    return result


def ablate_channel(inputs, channel):
    result = inputs.copy()
    result[:, channel, :] = 0.0
    return result


def role_names():
    per_channel = ("mean", "std", "rms", "mean_abs_change", "trend", "frequency_peak")
    return [f"channel_{channel}_{role}" for channel in range(CHANNELS) for role in per_channel] + ["global_rms", "global_mean_abs_change", "cross_channel_std"]


def raw_role_features(inputs):
    values = np.asarray(inputs)
    features = []
    for channel in values:
        changes = np.diff(channel)
        spectrum = np.abs(np.fft.rfft(channel - channel.mean()))[1:]
        features.extend((channel.mean(), channel.std(), np.sqrt(np.mean(channel * channel)), np.mean(np.abs(changes)), channel[-1] - channel[0], spectrum.max() / max(1, len(channel))))
    features.extend((np.sqrt(np.mean(values * values)), np.mean(np.abs(np.diff(values, axis=1))), values.std(axis=0).mean()))
    return np.asarray(features)


def _window_role_values(signal):
    centered = signal - signal.mean()
    changes = np.diff(signal)
    spectrum = np.abs(np.fft.rfft(centered))[1:]
    peaks = ((signal[1:-1] > signal[:-2]) & (signal[1:-1] > signal[2:])).sum()
    zero_crossings = (centered[:-1] * centered[1:] < 0).sum()
    periodicity = np.mean(centered[:-1] * centered[1:]) / (signal.std() ** 2 + 1e-12)
    return (signal.mean(), signal.std(), np.sqrt(np.mean(signal * signal)), np.mean(np.abs(changes)), signal[-1] - signal[0], spectrum.max() / max(1, len(signal)), peaks / len(signal), zero_crossings / len(signal), np.max(np.abs(signal)), np.max(np.abs(changes)), np.mean(np.abs(changes) < 0.1), periodicity)


def expanded_role_names():
    names = [f"window_{window}_channel_{channel}_{role}" for window in range(WINDOWS) for channel in range(CHANNELS) for role in WINDOW_ROLES]
    names += [f"window_{window}_{role}" for window in range(WINDOWS) for role in CROSS_WINDOW_ROLES]
    names += list(GLOBAL_ROLES)
    return names


def expanded_role_features(inputs):
    values = np.asarray(inputs)
    features = []
    for window in range(WINDOWS):
        current = values[:, window * WINDOW_STEPS:(window + 1) * WINDOW_STEPS]
        for channel in current:
            features.extend(_window_role_values(channel))
        changes = np.diff(current, axis=1)
        features.extend((np.sqrt(np.mean(current[0:3] ** 2)), np.sqrt(np.mean(current[3:6] ** 2)), np.sqrt(np.mean(current[6:9] ** 2)), np.mean(current[0:3] * current[3:6]), current.std(axis=0).mean(), np.mean(np.abs(changes), axis=0).mean()))
    changes = np.diff(values, axis=1)
    features.extend((np.sqrt(np.mean(values * values)), np.mean(np.abs(changes)), np.mean(np.abs(changes) < 0.1)))
    return np.asarray(features)


def expanded_role_matrix(inputs):
    return np.vstack([expanded_role_features(row) for row in inputs])


def temporal_role_names():
    names = []
    for window in range(WINDOWS):
        names.extend(f"window_{window}_{role}" for role in WINDOW_ROLES)
        names.extend(f"window_{window}_{role}" for role in CROSS_WINDOW_ROLES)
    return names + list(GLOBAL_ROLES)


def temporal_role_features(inputs):
    raw = expanded_role_features(inputs)
    per_channel_count = WINDOWS * CHANNELS * len(WINDOW_ROLES)
    per_channel = raw[:per_channel_count].reshape(WINDOWS, CHANNELS, len(WINDOW_ROLES)).mean(axis=1)
    cross = raw[per_channel_count:per_channel_count + WINDOWS * len(CROSS_WINDOW_ROLES)].reshape(WINDOWS, len(CROSS_WINDOW_ROLES))
    global_values = raw[-len(GLOBAL_ROLES):]
    return np.concatenate([np.column_stack((per_channel, cross)).ravel(), global_values])


def temporal_role_matrix(inputs):
    return np.vstack([temporal_role_features(row) for row in inputs])


def role_matrix(inputs):
    return np.vstack([raw_role_features(row) for row in inputs])


def train_role_model(x, y, epochs=300, learning_rate=0.05, seed=7):
    features = role_matrix(x)
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
        weights -= learning_rate * features.T @ error
        bias -= learning_rate * error.sum(axis=0)
    return {"weights": weights, "bias": bias, "mean": mean, "scale": scale}


def train_role_hidden_model(x, y, hidden=FILTERS, epochs=400, learning_rate=0.03, seed=7):
    features = role_matrix(x)
    mean, scale = features.mean(axis=0), features.std(axis=0)
    scale[scale == 0] = 1.0
    features = (features - mean) / scale
    rng = np.random.default_rng(seed)
    hidden_weights = rng.normal(0, np.sqrt(2 / features.shape[1]), (features.shape[1], hidden))
    hidden_bias = np.zeros(hidden)
    output = rng.normal(0, np.sqrt(2 / hidden), (hidden, CLASSES))
    output_bias = np.zeros(CLASSES)
    for _ in range(epochs):
        hidden_pre = features @ hidden_weights + hidden_bias
        hidden_values = np.maximum(hidden_pre, 0.0)
        probabilities = softmax(hidden_values @ output + output_bias)
        error = probabilities.copy()
        error[np.arange(len(y)), y] -= 1.0
        error /= len(y)
        output_gradient = hidden_values.T @ error
        output_bias_gradient = error.sum(axis=0)
        hidden_error = (error @ output.T) * (hidden_pre > 0)
        hidden_weights_gradient = features.T @ hidden_error
        hidden_bias_gradient = hidden_error.sum(axis=0)
        hidden_weights -= learning_rate * hidden_weights_gradient
        hidden_bias -= learning_rate * hidden_bias_gradient
        output -= learning_rate * output_gradient
        output_bias -= learning_rate * output_bias_gradient
    return {"hidden_weights": hidden_weights, "hidden_bias": hidden_bias, "output": output, "output_bias": output_bias, "mean": mean, "scale": scale}


def role_predict(model, inputs):
    features = (role_matrix(inputs) - model["mean"]) / model["scale"]
    return np.argmax(features @ model["weights"] + model["bias"], axis=1)


def role_hidden_predict(model, inputs):
    features = (role_matrix(inputs) - model["mean"]) / model["scale"]
    hidden = np.maximum(features @ model["hidden_weights"] + model["hidden_bias"], 0.0)
    return np.argmax(hidden @ model["output"] + model["output_bias"], axis=1)


def role_accuracy(model, inputs, targets):
    return float(np.mean(role_predict(model, inputs) == targets))


def role_hidden_accuracy(model, inputs, targets):
    return float(np.mean(role_hidden_predict(model, inputs) == targets))


def grouped_role_matrix(inputs):
    raw = expanded_role_matrix(inputs)
    expanded = raw[:, :WINDOWS * CHANNELS * len(WINDOW_ROLES)].reshape(len(inputs), WINDOWS, CHANNELS, len(WINDOW_ROLES))
    grouped = np.column_stack((
        (expanded[:, :, :, 0].mean(axis=(1, 2)) + expanded[:, :, :, 2].mean(axis=(1, 2))) / 2,
        expanded[:, :, :, 1].mean(axis=(1, 2)),
        expanded[:, :, :, 3].mean(axis=(1, 2)),
        np.abs(expanded[:, :, :, 4]).mean(axis=(1, 2)),
        expanded[:, :, :, 5].mean(axis=(1, 2)),
        expanded[:, :, :, 6].mean(axis=(1, 2)),
        expanded[:, :, :, 7].mean(axis=(1, 2)),
        expanded[:, :, :, 8].mean(axis=(1, 2)),
        expanded[:, :, :, 9].mean(axis=(1, 2)),
        expanded[:, :, :, 10].mean(axis=(1, 2)),
        expanded[:, :, :, 11].mean(axis=(1, 2)),
    ))
    cross = raw[:, -(WINDOWS * len(CROSS_WINDOW_ROLES)):]
    return np.column_stack((grouped, cross.reshape(len(inputs), WINDOWS, len(CROSS_WINDOW_ROLES)).mean(axis=1).mean(axis=1)))


def grouped_role_names():
    return ("level_energy", "variability", "change", "trend", "frequency", "peak_rate", "zero_crossing_rate", "amplitude", "abrupt_change", "stillness", "periodicity", "sensor_relation")


def train_expanded_role_model(x, y, epochs=300, learning_rate=0.05, seed=7):
    features = grouped_role_matrix(x)
    mean, scale = features.mean(axis=0), features.std(axis=0)
    scale[scale == 0] = 1.0
    features = (features - mean) / scale
    rng = np.random.default_rng(seed)
    output = rng.normal(0, 0.05, (features.shape[1], CLASSES))
    output_bias = np.zeros(CLASSES)
    for _ in range(epochs):
        probabilities = softmax(features @ output + output_bias)
        error = probabilities.copy()
        error[np.arange(len(y)), y] -= 1.0
        error /= len(y)
        output -= learning_rate * features.T @ error
        output_bias -= learning_rate * error.sum(axis=0)
    return {"output": output, "output_bias": output_bias, "mean": mean, "scale": scale}


def expanded_role_predict(model, inputs):
    features = (grouped_role_matrix(inputs) - model["mean"]) / model["scale"]
    return np.argmax(features @ model["output"] + model["output_bias"], axis=1)


def expanded_role_accuracy(model, inputs, targets):
    return float(np.mean(expanded_role_predict(model, inputs) == targets))


def train_temporal_role_model(x, y, epochs=300, learning_rate=0.05, seed=7):
    features = temporal_role_matrix(x)
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
        weights -= learning_rate * features.T @ error
        bias -= learning_rate * error.sum(axis=0)
    return {"weights": weights, "bias": bias, "mean": mean, "scale": scale}


def temporal_role_predict(model, inputs):
    features = (temporal_role_matrix(inputs) - model["mean"]) / model["scale"]
    return np.argmax(features @ model["weights"] + model["bias"], axis=1)


def temporal_role_accuracy(model, inputs, targets):
    return float(np.mean(temporal_role_predict(model, inputs) == targets))


def correlation_matrix(roles, pooled):
    roles = (roles - roles.mean(axis=0)) / np.maximum(roles.std(axis=0), 1e-12)
    pooled = (pooled - pooled.mean(axis=0)) / np.maximum(pooled.std(axis=0), 1e-12)
    return roles.T @ pooled / len(roles)


def analyze(model, inputs, targets):
    activations, pooled, _ = cnn_forward(model, inputs)
    baseline = accuracy(model, inputs, targets)
    ablations = []
    for index in range(FILTERS):
        score = accuracy(ablate_filter(model, index), inputs, targets)
        ablations.append({"filter": index, "accuracy_after_removal": score, "accuracy_drop": baseline - score})
    top_pair = [row["filter"] for row in sorted(ablations, key=lambda row: row["accuracy_drop"], reverse=True)[:2]]
    pair_score = accuracy(ablate_filters(model, top_pair), inputs, targets)
    roles = role_matrix(inputs)
    correlations = correlation_matrix(roles, pooled)
    best_roles = [int(np.argmax(np.abs(correlations[:, index]))) for index in range(FILTERS)]
    return {
        "baseline_accuracy": baseline,
        "filter_ablations": sorted(ablations, key=lambda row: row["accuracy_drop"], reverse=True),
        "filter_activation_mean": pooled.mean(axis=0).tolist(),
        "filter_role_matches": [{"filter": index, "role": role_names()[best_roles[index]], "correlation": float(correlations[best_roles[index], index])} for index in range(FILTERS)],
        "top_pair": {"filters": top_pair, "accuracy_after_removal": pair_score, "accuracy_drop": baseline - pair_score},
        "role_names": role_names(),
    }


def json_model(model):
    return {key: value.tolist() for key, value in model.items()}


def run(root, data_dir):
    root, data_dir = Path(root), Path(data_dir)
    data = load_raw(data_dir)
    baseline = train_model(data["train_x"], data["train_y"])
    roles = train_role_model(data["train_x"], data["train_y"])
    role_hidden = train_role_hidden_model(data["train_x"], data["train_y"])
    expanded_roles = train_expanded_role_model(data["train_x"], data["train_y"])
    temporal_roles = train_temporal_role_model(data["train_x"], data["train_y"])
    baseline_predictions = predict(baseline, data["test_x"])
    role_predictions = role_predict(roles, data["test_x"])
    role_hidden_predictions = role_hidden_predict(role_hidden, data["test_x"])
    expanded_role_predictions = expanded_role_predict(expanded_roles, data["test_x"])
    temporal_role_predictions = temporal_role_predict(temporal_roles, data["test_x"])
    noisy = data["test_x"] + np.random.default_rng(19).normal(0, 0.05, data["test_x"].shape)
    scaled = data["test_x"] * 1.1
    analysis = analyze(baseline, data["test_x"], data["test_y"])
    seed_stability = []
    for seed in (11, 19):
        alternate = train_model(data["train_x"], data["train_y"], seed=seed)
        seed_stability.append({"seed": seed, "train": accuracy(alternate, data["train_x"], data["train_y"]), "test": accuracy(alternate, data["test_x"], data["test_y"])})
    results = {
        "architecture": {"input": [CHANNELS, STEPS], "conv_filters": FILTERS, "kernel": KERNEL, "output": CLASSES},
        "parameters": count_parameters(),
        "baseline": {"train": accuracy(baseline, data["train_x"], data["train_y"]), "test": accuracy(baseline, data["test_x"], data["test_y"]), "noise_0.05": accuracy(baseline, noisy, data["test_y"]), "scale_1.1": accuracy(baseline, scaled, data["test_y"])},
        "redesigned": {"train": role_accuracy(roles, data["train_x"], data["train_y"]), "test": role_accuracy(roles, data["test_x"], data["test_y"]), "noise_0.05": role_accuracy(roles, noisy, data["test_y"]), "scale_1.1": role_accuracy(roles, scaled, data["test_y"]), "parameters": int(roles["weights"].size + roles["bias"].size), "roles": role_names()},
        "role_hidden": {"train": role_hidden_accuracy(role_hidden, data["train_x"], data["train_y"]), "test": role_hidden_accuracy(role_hidden, data["test_x"], data["test_y"]), "noise_0.05": role_hidden_accuracy(role_hidden, noisy, data["test_y"]), "scale_1.1": role_hidden_accuracy(role_hidden, scaled, data["test_y"]), "parameters": int(role_hidden["hidden_weights"].size + role_hidden["hidden_bias"].size + role_hidden["output"].size + role_hidden["output_bias"].size), "hidden_nodes": FILTERS},
        "expanded_roles": {"train": expanded_role_accuracy(expanded_roles, data["train_x"], data["train_y"]), "test": expanded_role_accuracy(expanded_roles, data["test_x"], data["test_y"]), "noise_0.05": expanded_role_accuracy(expanded_roles, noisy, data["test_y"]), "scale_1.1": expanded_role_accuracy(expanded_roles, scaled, data["test_y"]), "parameters": int(expanded_roles["output"].size + expanded_roles["output_bias"].size), "calculation_count": len(expanded_role_names()), "role_nodes": len(grouped_role_names())},
        "temporal_roles": {"train": temporal_role_accuracy(temporal_roles, data["train_x"], data["train_y"]), "test": temporal_role_accuracy(temporal_roles, data["test_x"], data["test_y"]), "noise_0.05": temporal_role_accuracy(temporal_roles, noisy, data["test_y"]), "scale_1.1": temporal_role_accuracy(temporal_roles, scaled, data["test_y"]), "parameters": int(temporal_roles["weights"].size + temporal_roles["bias"].size), "role_values": len(temporal_role_names())},
        "teacher_student_agreement": float(np.mean(baseline_predictions == role_predictions)),
        "teacher_role_hidden_agreement": float(np.mean(baseline_predictions == role_hidden_predictions)),
        "teacher_expanded_role_agreement": float(np.mean(baseline_predictions == expanded_role_predictions)),
        "teacher_temporal_role_agreement": float(np.mean(baseline_predictions == temporal_role_predictions)),
        "seed_stability": [{"seed": 7, "train": accuracy(baseline, data["train_x"], data["train_y"]), "test": accuracy(baseline, data["test_x"], data["test_y"])}] + seed_stability,
        "channel_ablation": [{"channel": channel, "accuracy_after_zeroing": accuracy(baseline, ablate_channel(data["test_x"], channel), data["test_y"])} for channel in range(CHANNELS)],
        "analysis": analysis,
    }
    np.savez(root / "baseline_model.npz", **baseline)
    np.savez(root / "role_model.npz", **roles)
    np.savez(root / "role_hidden_model.npz", **role_hidden)
    np.savez(root / "expanded_role_model.npz", **expanded_roles)
    np.savez(root / "temporal_role_model.npz", **temporal_roles)
    results["model_file_bytes"] = {"baseline": (root / "baseline_model.npz").stat().st_size, "redesigned": (root / "role_model.npz").stat().st_size}
    results["model_file_bytes"]["role_hidden"] = (root / "role_hidden_model.npz").stat().st_size
    results["model_file_bytes"]["expanded_roles"] = (root / "expanded_role_model.npz").stat().st_size
    results["model_file_bytes"]["temporal_roles"] = (root / "temporal_role_model.npz").stat().st_size
    (root / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    (root / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    return results


if __name__ == "__main__":
    here = Path(__file__).parent
    print(json.dumps(run(here, here.parent / "data" / "UCI HAR Dataset"), indent=2))
