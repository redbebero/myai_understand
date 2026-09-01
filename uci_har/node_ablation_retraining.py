"""Test whether low-impact node removal is harmful, neutral, or recoverable."""
import json
from pathlib import Path
import numpy as np
from .uci_har_experiment import ablate_hidden2, accuracy, baseline_forward, load_data, train_baseline, _adam_update
from .node_ablation_reanalysis import SEEDS, VAL_SUBJECTS, node_drop_ranking, paired_metrics


def finetune_masked(model, x, y, frozen_nodes, epochs=30, seed=0):
    model = {name: value.copy() for name, value in model.items()}
    frozen = np.zeros(32, dtype=bool); frozen[list(frozen_nodes)] = True
    rng = np.random.default_rng(seed + 1001)
    moments = {name: [np.zeros_like(value), np.zeros_like(value)] for name, value in model.items()}
    step = 0
    for _ in range(epochs):
        for indices in np.array_split(rng.permutation(len(x)), max(1, len(x) // 128)):
            bx, by = x[indices], y[indices]
            h1, h2, p = baseline_forward(model, bx)
            error = p.copy(); error[np.arange(len(by)), by] -= 1.0; error /= len(by)
            dh2 = (error @ model["w3"].T) * (h2 > 0)
            gradients = {"w3": h2.T @ error, "b3": error.sum(0), "w2": h1.T @ dh2, "b2": dh2.sum(0)}
            dh1 = (dh2 @ model["w2"].T) * (h1 > 0)
            gradients.update({"w1": bx.T @ dh1, "b1": dh1.sum(0)})
            step += 1; _adam_update(model, gradients, moments, step, 0.001)
            model["b2"][frozen] = 0.0; model["w2"][:, frozen] = 0.0; model["w3"][frozen, :] = 0.0
    return model


def geometry(model, x, y):
    _, h, _ = baseline_forward(model, x)
    centers = np.vstack([h[y == c].mean(0) for c in range(6)])
    within = float(np.mean([np.mean((h[y == c] - centers[c]) ** 2) for c in range(6)]))
    between = float(np.mean([(centers[i] - centers[j]) @ (centers[i] - centers[j]) for i in range(6) for j in range(i)]))
    return {"within_mse": within, "between_centroid_distance": between, "separation_ratio": between / within if within else 0.0}


def run(root):
    data = load_data(Path(root) / "UCI HAR Dataset")
    fit = ~np.isin(data["subject_train"], VAL_SUBJECTS); val = ~fit
    runs=[]
    for seed in SEEDS:
        base = train_baseline(data["train_x"][fit], data["train_y"][fit], seed=seed)
        selected = [r["node"] for r in node_drop_ranking(base, data["train_x"][val], data["train_y"][val])[:2]]
        post = ablate_hidden2(ablate_hidden2(base, selected[0]), selected[1])
        tuned = finetune_masked(post, data["train_x"][fit], data["train_y"][fit], selected, seed=seed)
        rows={"seed":seed,"selected_nodes":selected}
        for name, model in (("baseline",base),("posthoc",post),("finetuned",tuned)):
            rows[name]={"validation_accuracy":accuracy(model,data["train_x"][val],data["train_y"][val]),"test_accuracy":accuracy(model,data["test_x"],data["test_y"]),"test_geometry":geometry(model,data["test_x"],data["test_y"])}
        rows["posthoc_vs_baseline_test"]=paired_metrics(np.argmax(baseline_forward(base,data["test_x"])[2],1),np.argmax(baseline_forward(post,data["test_x"])[2],1),data["test_y"])
        rows["finetuned_vs_baseline_test"]=paired_metrics(np.argmax(baseline_forward(base,data["test_x"])[2],1),np.argmax(baseline_forward(tuned,data["test_x"])[2],1),data["test_y"])
        runs.append(rows)
    result={"design":{"seeds":list(SEEDS),"validation_subjects":list(VAL_SUBJECTS),"fine_tune_epochs":30,"selection":"two lowest individual validation ablation drops","test_used_for_selection":False},"runs":runs}
    for key in ("baseline","posthoc","finetuned"):
        result[key+"_mean_test_accuracy"]=float(np.mean([r[key]["test_accuracy"] for r in runs]))
    (Path(root)/"node_ablation_retraining_results.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:result[k] for k in result if k.endswith("mean_test_accuracy")},indent=2))

if __name__ == "__main__": run(Path(__file__).parent)
