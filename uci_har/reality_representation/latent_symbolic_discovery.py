"""Find a small learned latent representation and map it back to raw signals."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiment import VAL_SUBJECTS, load_raw, standardize
from bottleneck_gate_experiment import forward
from auto_structure_discovery import candidate_expressions
from advanced_input_trace import lasso_explain

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "results" / "bottleneck_gate"
OUT = HERE / "results" / "bottleneck_gate" / "latent_symbolic_results.json"
CHANNELS = ("body_acc_x", "body_acc_y", "body_acc_z", "body_gyro_x", "body_gyro_y", "body_gyro_z", "total_acc_x", "total_acc_y", "total_acc_z")


def load_model(path):
    z = np.load(path); return {key: z[key] for key in z.files}


def latent(model, x):
    return forward(model, x, training=True)[1][-1]


def latent_influence(model, x, dimension):
    flat = x.reshape(len(x), -1)
    _, cache = forward(model, x, training=True)
    flat, gate, gated, z1, h1, z2, h2, _ = cache
    d1 = z1 > 0; d2 = z2 > 0
    vector = np.zeros((len(x), flat.shape[1]))
    for i in range(len(x)):
        vector[i] = gate * ((model["w1"] * d1[i][None, :]) @ model["w2"] @ (d2[i] * model["w3"][:, dimension]))
    return vector.reshape(len(x), 9, 128)


def regions_from_influence(influence, q=.85, width=16):
    regions = []
    for c in range(9):
        strength = influence[:, c].mean(0)
        threshold = np.quantile(strength, q); mask = strength >= threshold; start = None
        for t, active in enumerate(np.r_[mask, False]):
            if active and start is None: start = t
            elif not active and start is not None:
                if t - start >= 8: regions.append({"channel": c, "start": start, "end": t, "strength": float(strength[start:t].mean())})
                start = None
        if not any(r["channel"] == c for r in regions):
            means = np.convolve(strength, np.ones(width) / width, mode="valid"); s = int(np.argmax(means))
            regions.append({"channel": c, "start": s, "end": s + width, "strength": float(means[s])})
    return sorted(regions, key=lambda r: r["strength"], reverse=True)[:8]


def run(seeds=(7, 11, 19, 23, 31), variants=("flat", "wide"), k=4):
    train_x, _, subjects, test_x, _, _ = load_raw(); train_x, test_x = standardize(train_x, test_x)
    fitting = ~np.isin(subjects, VAL_SUBJECTS); results = []; z_all = []
    for variant in variants:
        for seed in seeds:
            model = load_model(MODEL_DIR / f"model_{variant}_k{k}_seed{seed}.npz")
            z_fit, z_val = latent(model, train_x[fitting]), latent(model, train_x[~fitting]); z_all.append(z_val)
            latent_results = []
            for dimension in range(k):
                regions = regions_from_influence(np.abs(latent_influence(model, train_x[fitting], dimension)))
                fit_expr, names, _ = candidate_expressions(train_x[fitting], regions); val_expr, _, _ = candidate_expressions(train_x[~fitting], regions)
                explanation = lasso_explain(z_fit[:, dimension], z_val[:, dimension], fit_expr, val_expr, names, iterations=40)
                latent_results.append({"dimension": dimension, "regions": regions, "expressions": explanation})
            results.append({"variant": variant, "seed": seed, "latent_results": latent_results, "test_accuracy": None})
    correlations = []
    for i in range(len(z_all)):
        for j in range(i + 1, len(z_all)):
            a, b = z_all[i] - z_all[i].mean(0), z_all[j] - z_all[j].mean(0)
            singular = np.linalg.svd(a.T @ b, compute_uv=False)
            correlations.append(float(singular[0] / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12)))
    result = {"bottleneck_k": k, "latent_subspace_proxy_mean": float(np.mean(correlations)), "latent_subspace_proxy_min": float(np.min(correlations)), "runs": results}
    OUT.write_text(json.dumps(result, indent=2)); return result


if __name__ == "__main__":
    result = run()
    for row in result["runs"]:
        print(row["variant"], row["seed"], [(x["dimension"], round(x["expressions"]["validation_r2"], 3), x["expressions"]["features"][0]["name"] if x["expressions"]["features"] else None) for x in row["latent_results"]])
