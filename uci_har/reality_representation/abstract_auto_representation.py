"""Aggregate AI-discovered formulas into operation-level concepts."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from experiment import VAL_SUBJECTS, load_raw, standardize
from auto_structure_discovery import candidate_expressions
from family_classifier_evaluation import train_linear
from improved_auto_comparison import evaluate_expression

HERE=Path(__file__).resolve().parent
INPUT=HERE/"results"/"bottleneck_gate"/"latent_symbolic_results.json"
OUT=HERE/"results"/"advanced_trace"/"abstract_auto_results.json"

def operation(name):
    if name.startswith("mean(square("): return "movement_energy"
    if name.startswith("mean(abs(diff("): return "temporal_change"
    if name.startswith("max("): return "local_range"
    if name.startswith("slope("): return "trend"
    if name.startswith("corr(") or "*" in name: return "sensor_coordination"
    return "sensor_level"

def run():
    tr,y,s,te,yt,_=load_raw(); tr,te=standardize(tr,te); fitmask=~np.isin(s,VAL_SUBJECTS); rows=[]
    auto=json.loads(INPUT.read_text())
    for source in auto["runs"]:
        regions=[]
        for latent in source["latent_results"]: regions.extend(latent["regions"])
        unique=[]; seen=set()
        for r in regions:
            key=(r["channel"],r["start"],r["end"])
            if key not in seen: seen.add(key); unique.append(r)
        _,names,_=candidate_expressions(tr[fitmask],unique)
        fit=np.column_stack([evaluate_expression(n,tr[fitmask]) for n in names]); test=np.column_stack([evaluate_expression(n,te) for n in names])
        fit=np.nan_to_num(fit); test=np.nan_to_num(test)
        groups={k:[] for k in ["movement_energy","temporal_change","local_range","trend","sensor_coordination","sensor_level"]}
        for i,n in enumerate(names): groups[operation(n)].append(i)
        concepts=[]; concept_names=[]
        for group,indices in groups.items():
            if not indices: continue
            mean=fit[:,indices].mean(0); scale=fit[:,indices].std(0); scale[scale<1e-10]=1
            concepts.append(np.mean((fit[:,indices]-mean)/scale,axis=1)); concept_names.append(group)
        X=np.column_stack(concepts); T=np.column_stack([np.mean((test[:,idx] - fit[:,idx].mean(0))/np.where(fit[:,idx].std(0)<1e-10,1,fit[:,idx].std(0)),axis=1) for idx in groups.values() if idx])
        w,b=train_linear(X,y[fitmask],source["seed"],epochs=300)
        rows.append({"variant":source["variant"],"seed":source["seed"],"candidate_count":len(names),"concepts":concept_names,"concept_sizes":{k:len(v) for k,v in groups.items() if v},"test_accuracy":float(np.mean(np.argmax(T@w+b,1)==yt))})
    result={"aggregation":"standardized mean of AI-discovered expressions grouped by mathematical operation","runs":rows}; OUT.write_text(json.dumps(result,indent=2)); return result

if __name__=="__main__":
    result=run(); print(result["runs"])
