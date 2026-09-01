"""Raw-sensor representation reverse-engineering experiment.

The model receives every raw inertial channel. All interpretation choices happen
on the validation split; the official test subjects are used only at the end.
"""
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "UCI HAR Dataset"
OUT = Path(__file__).resolve().parent / "results"
CHANNELS = (
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
)
LABELS = ("walking", "walking_upstairs", "walking_downstairs", "sitting", "standing", "laying")
VAL_SUBJECTS = (1, 3, 5, 6, 7, 8)


def load_raw(root: Path = ROOT):
    def matrix(split, stem):
        return np.loadtxt(root / split / "Inertial Signals" / f"{stem}_{split}.txt", dtype=np.float64)

    def labels(split):
        return np.loadtxt(root / split / f"y_{split}.txt", dtype=int) - 1

    def subjects(split):
        return np.loadtxt(root / split / f"subject_{split}.txt", dtype=int)

    def stack(split):
        return np.stack([matrix(split, name) for name in CHANNELS], axis=1)

    return stack("train"), labels("train"), subjects("train"), stack("test"), labels("test"), subjects("test")


def standardize(train, other):
    mean, scale = train.mean(axis=(0, 2), keepdims=True), train.std(axis=(0, 2), keepdims=True)
    scale[scale == 0] = 1
    return (train - mean) / scale, (other - mean) / scale

def relu(x):
    return np.maximum(x, 0)


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    p = np.exp(x)
    return p / p.sum(axis=1, keepdims=True)


def new_model(seed, input_size=9 * 128, hidden=32, variant="flat"):
    rng = np.random.default_rng(seed)
    width = 64 if variant == "flat" else 96
    return {
        "w1": rng.normal(0, np.sqrt(2 / input_size), (input_size, width)),
        "b1": np.zeros(width), "w2": rng.normal(0, np.sqrt(2 / width), (width, hidden)),
        "b2": np.zeros(hidden), "w3": rng.normal(0, np.sqrt(2 / hidden), (hidden, 6)),
        "b3": np.zeros(6),
    }

def forward(model, x):
    flat = x.reshape(len(x), -1)
    h1 = relu(flat @ model["w1"] + model["b1"])
    h = relu(h1 @ model["w2"] + model["b2"])
    return h1, h, softmax(h @ model["w3"] + model["b3"])

def train(x, y, seed, variant="flat", epochs=45, batch_size=128, learning_rate=0.001):
    model, rng = new_model(seed, variant=variant), np.random.default_rng(seed + 1000)
    moments = {k: [np.zeros_like(v), np.zeros_like(v)] for k, v in model.items()}
    step = 0
    for _ in range(epochs):
        for idx in np.array_split(rng.permutation(len(x)), max(1, len(x) // batch_size)):
            batch, target = x[idx], y[idx]
            h1, h, p = forward(model, batch)
            e = p.copy(); e[np.arange(len(target)), target] -= 1; e /= len(target)
            dh = e @ model["w3"].T * (h > 0)
            dh1 = dh @ model["w2"].T * (h1 > 0)
            gradients = {"w3": h.T @ e, "b3": e.sum(0), "w2": h1.T @ dh,
                         "b2": dh.sum(0), "w1": batch.reshape(len(batch), -1).T @ dh1,
                         "b1": dh1.sum(0)}
            step += 1
            for name, g in gradients.items():
                m, v = moments[name]
                m[:] = .9 * m + .1 * g; v[:] = .999 * v + .001 * g * g
                model[name] -= learning_rate * (m / (1 - .9 ** step)) / (np.sqrt(v / (1 - .999 ** step)) + 1e-8)
    return model


def accuracy(model, x, y):
    return float(np.mean(forward(model, x)[2].argmax(1) == y))


def task_directions(model, k):
    # Directions in the hidden representation visible to the classifier.
    u, _, _ = np.linalg.svd(model["w3"], full_matrices=False)
    return u[:, :k]


def projected_accuracy(model, x, y, k):
    h = forward(model, x)[1]
    w = task_directions(model, k)
    projected = (h @ w) @ w.T
    return float(np.mean((projected @ model["w3"] + model["b3"]).argmax(1) == y))


def break_relations(x, seed=0):
    rng = np.random.default_rng(seed)
    result = x.copy()
    # Preserve every channel's marginal values; destroy cross-channel pairing.
    for i in range(len(result)):
        for channel in range(result.shape[1]):
            result[i, channel] = result[i, channel, rng.permutation(result.shape[2])]
    return result

def dictionary(x):
    features, names = [], []
    for c, name in enumerate(CHANNELS):
        signal = x[:, c, :]
        features += [signal.mean(1), signal.std(1), np.mean(signal * signal, 1), np.mean(np.diff(signal, axis=1), 1)]
        names += [f"{name}:level", f"{name}:variation", f"{name}:energy", f"{name}:slope"]
    for a in range(9):
        for b in range(a + 1, 9):
            features.append(np.mean(x[:, a, :] * x[:, b, :], 1))
            names.append(f"{CHANNELS[a]}×{CHANNELS[b]}:coupling")
    values = np.asarray(features).T
    values = (values - values.mean(0)) / np.where(values.std(0) == 0, 1, values.std(0))
    return values, tuple(names)


def top_sensor_explanations(model, fit_x, heldout_x, top=5):
    fit_h, heldout_h = forward(model, fit_x)[1], forward(model, heldout_x)[1]
    fit_d, fit_names = dictionary(fit_x)
    heldout_d, _ = dictionary(heldout_x)
    fit_scores = np.abs(np.corrcoef(fit_h.T, fit_d.T)[:fit_h.shape[1], fit_h.shape[1]:])
    heldout_scores = np.abs(np.corrcoef(heldout_h.T, heldout_d.T)[:heldout_h.shape[1], heldout_h.shape[1]:])
    result = []
    for axis in range(fit_h.shape[1]):
        order = np.argsort(fit_scores[axis])[::-1][:top]
        result.append({"latent_axis": axis, "features": [
            {"name": fit_names[i], "fit_correlation": float(fit_scores[axis, i]),
             "heldout_correlation": float(heldout_scores[axis, i])} for i in order]})
    return result


def intervene_coupling(x, feature_name, seed=0, mode="shift"):
    match = re.match(r"([^×]+)×([^:]+):coupling", feature_name)
    if not match:
        return break_relations(x, seed)
    b = CHANNELS.index(match.group(2))
    result = x.copy()
    rng = np.random.default_rng(seed)
    for i in range(len(result)):
        lag = int(rng.integers(1, result.shape[2])) if mode == "shift" else None
        order = np.roll(np.arange(result.shape[2]), lag) if mode == "shift" else rng.permutation(result.shape[2])
        result[i, b] = result[i, b, order]
    return result
def human_abstraction(feature_names):
    text = " ".join(feature_names)
    if "coupling" in text and "gyro" in text and "acc" in text:
        return "linear-motion and rotation coupling"
    if "coupling" in text:
        return "cross-channel coordination"
    if "variation" in text or "slope" in text:
        return "temporal change pattern"
    if "energy" in text:
        return "movement intensity"
    return "sensor level or orientation pattern"


def direction_effects(model, x, y):
    h = forward(model, x)[1]
    directions = task_directions(model, h.shape[1])
    logits = h @ model["w3"] + model["b3"]
    effects = []
    for axis in range(directions.shape[1]):
        component = np.outer(h @ directions[:, axis], directions[:, axis])
        without = (h - component) @ model["w3"] + model["b3"]
        pair_effects = {}
        for a in range(6):
            for b in range(a + 1, 6):
                before = logits[:, a] - logits[:, b]
                after = without[:, a] - without[:, b]
                pair_effects[f"{LABELS[a]} vs {LABELS[b]}"] = float(np.mean(np.abs(before - after)))
        effects.append(pair_effects)
    return effects
def intervention_effect(model, x, y, feature_name, seed=0):
    before = forward(model, x)[2]
    after = forward(model, intervene_coupling(x, feature_name, seed, mode="shift"))[2]
    pair_change = {}
    for a in range(6):
        for b in range(a + 1, 6):
            key = f"{LABELS[a]} vs {LABELS[b]}"
            pair_change[key] = float(np.mean(np.abs((before[:, a] - before[:, b]) - (after[:, a] - after[:, b]))))
    strongest = max(pair_change, key=pair_change.get)
    return {"feature": feature_name, "true_class_probability_change": float(np.mean(np.abs(after[np.arange(len(y)), y] - before[np.arange(len(y)), y]))),
            "prediction_change_rate": float(np.mean(after.argmax(1) != before.argmax(1))),
            "activity_pair_margin_change": pair_change, "strongest_activity_pair": strongest}

def run(seeds=(7, 11, 19, 23, 31)):
    train_x, train_y, train_subject, test_x, test_y, test_subject = load_raw()
    train_x, test_x = standardize(train_x, test_x)
    val = np.isin(train_subject, VAL_SUBJECTS)
    fit = ~val
    OUT.mkdir(exist_ok=True)
    all_results, explanations = [], []
    for variant in ("flat", "wide"):
        for seed in seeds:
            model = train(train_x[fit], train_y[fit], seed, variant=variant)
            base = accuracy(model, test_x, test_y)
            relation = accuracy(model, break_relations(test_x, seed), test_y)
            curve = {str(k): projected_accuracy(model, test_x, test_y, k) for k in (1, 2, 4, 8, 16, 32)}
            val_exp = top_sensor_explanations(model, train_x[fit], train_x[val], top=5)
            explanations.append({"variant": variant, "seed": seed, "axes": val_exp})
            candidates = [item["features"][0]["name"] for item in val_exp if item["features"] and "×" in item["features"][0]["name"]]
            chosen = candidates[0] if candidates else CHANNELS[0] + "×" + CHANNELS[1] + ":coupling"
            intervention = intervention_effect(model, test_x, test_y, chosen, seed + 2000)
            control = intervention_effect(model, test_x, test_y, CHANNELS[0] + "×" + CHANNELS[1] + ":coupling", seed + 3000)
            effects = direction_effects(model, test_x, test_y)
            cards = []
            for axis, (axis_info, pair_effect) in enumerate(zip(val_exp, effects)):
                features = axis_info["features"][:3]
                cards.append({"latent_direction": axis, "influence_by_activity_pair": pair_effect,
                              "required_sensor_information": features,
                              "human_abstraction": human_abstraction([item["name"] for item in features])})
            all_results.append({"variant": variant, "seed": seed, "test_accuracy": base,
                                "relation_broken_accuracy": relation, "relation_drop": base - relation,
                                "task_projection_accuracy": curve, "intervention": intervention,
                                "matched_control": control, "interpretation_cards": cards,
                                "test_subjects": sorted(set(test_subject.tolist()))})
    recurring = sorted(set(
        axis["features"][0]["name"]
        for exp in explanations for axis in exp["axes"] if axis["features"]
    ))
    result = {"protocol": {"input": "9 raw channels x 128 timesteps", "validation_subjects": VAL_SUBJECTS,
                            "relation_breaking": "within-sample channel permutation", "variants": ["flat", "wide"],
                            "seeds": list(seeds)},
              "runs": all_results, "explanations": explanations, "candidate_recurring_features": recurring}
    (OUT / "experiment_results.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    result = run()
    for row in result["runs"]:
        print(row["seed"], f"base={row['test_accuracy']:.4f}", f"relation_broken={row['relation_broken_accuracy']:.4f}",
              f"drop={row['relation_drop']:.4f}", row["task_projection_accuracy"])
