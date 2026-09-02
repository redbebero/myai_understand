"""Improve AI-only representation by selecting raw expressions for classification.

Regions remain AI-discovered. Expression selection is performed by held-out
validation classification accuracy rather than only latent reconstruction R².
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiment import VAL_SUBJECTS, load_raw, standardize
from auto_structure_discovery import candidate_expressions
from family_classifier_evaluation import train_linear

HERE = Path(__file__).resolve().parent
INPUT = HERE / "results" / "bottleneck_gate" / "latent_symbolic_results.json"
OUT = HERE / "results" / "advanced_trace" / "improved_auto_results.json"


def evaluate_expression(name, x):
    import re
    refs = [(int(c), int(a), int(b)) for c, a, b in re.findall(r"x\[(\d+),(\d+):(\d+)\]", name)]
    unique = []
    for ref in refs:
        if ref not in unique: unique.append(ref)
    signals = [x[:, c, a:b] for c, a, b in unique]
    if name.startswith("mean(square("): return np.mean(signals[0] ** 2, 1)
    if name.startswith("mean(abs(diff("): return np.mean(np.abs(np.diff(signals[0], axis=1)), 1)
    if name.startswith("max("): return np.ptp(signals[0], 1)
    if name.startswith("slope("): return np.polyfit(np.arange(signals[0].shape[1]), signals[0].T, 1)[0]
    if name.startswith("corr("):
        if len(signals) < 2: return signals[0].mean(1)
        a, b = signals; ac = a-a.mean(1,keepdims=True); bc=b-b.mean(1,keepdims=True); den=np.sqrt(np.sum(ac*ac,1)*np.sum(bc*bc,1)); return np.divide(np.sum(ac*bc,1),den,out=np.zeros(len(a)),where=den>1e-12)
    if len(signals) == 2:
        n=min(signals[0].shape[1],signals[1].shape[1]); return np.mean(signals[0][:,:n]*signals[1][:,:n],1)
    return signals[0].mean(1)


def accuracy_from_expression_subset(fit, val, y_fit, y_val, n_features, seed):
    mean, scale = fit.mean(0), fit.std(0); scale[scale < 1e-10] = 1
    fit=(fit-mean)/scale; val=(val-mean)/scale
    weights, bias = train_linear(fit[:, :n_features], y_fit, seed, epochs=300)
    return float(np.mean(np.argmax(val[:, :n_features] @ weights + bias, 1) == y_val))


def run(seeds=(7,11,19,23,31), variants=("flat","wide")):
    train_x, train_y, subjects, test_x, test_y, _ = load_raw()
    train_x, test_x = standardize(train_x, test_x)
    fitting=~np.isin(subjects,VAL_SUBJECTS); auto=json.loads(INPUT.read_text()); rows=[]
    for source in auto["runs"]:
        if source["variant"] not in variants or source["seed"] not in seeds: continue
        region_pool=[]
        for latent in source["latent_results"]: region_pool.extend(latent["regions"])
        unique_regions=[]
        for r in region_pool:
            key=(r["channel"],r["start"],r["end"])
            if key not in {(q["channel"],q["start"],q["end"]) for q in unique_regions}: unique_regions.append(r)
        fit_expr,names,_=candidate_expressions(train_x[fitting],unique_regions)
        fit=np.column_stack([evaluate_expression(n,train_x[fitting]) for n in names])
        val=np.column_stack([evaluate_expression(n,train_x[~fitting]) for n in names])
        test=np.column_stack([evaluate_expression(n,test_x) for n in names])
        order=np.argsort(-np.std(fit,axis=0))
        fit, val, test = (np.nan_to_num(a[:,order]) for a in (fit,val,test))
        names=tuple(names[i] for i in order)
        scores=[]
        for n in [5,10,20,40,min(80,len(names))]:
            if n>len(names): continue
            scores.append({"expression_count":n,"validation_accuracy":accuracy_from_expression_subset(fit,val,train_y[fitting],train_y[~fitting],n,source["seed"])})
        best=max(scores,key=lambda r:r["validation_accuracy"])
        mean,scale=fit.mean(0),fit.std(0); scale[scale<1e-10]=1
        weights,bias=train_linear(((fit-mean)/scale)[:,:best["expression_count"]],train_y[fitting],source["seed"],epochs=300)
        test_accuracy=float(np.mean(np.argmax(((test-mean)/scale)[:,:best["expression_count"]]@weights+bias,1)==test_y))
        rows.append({"variant":source["variant"],"seed":source["seed"],"candidate_count":len(names),"curves":scores,"best":best,"test_accuracy":test_accuracy,"selected_expressions":list(names[:best["expression_count"]])})
    result={"selection":"validation classification accuracy over AI-discovered raw regions","runs":rows}; OUT.write_text(json.dumps(result,indent=2)); return result

if __name__=="__main__":
    for r in run()["runs"]: print(r["variant"],r["seed"],r["best"])
