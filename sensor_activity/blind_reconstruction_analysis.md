# Blind Weight-Guided Reconstruction

## What was hidden

`blind_reconstruction.py` does not import `experiment.py` and does not read the
data-generation function. Its inputs are only:

```text
sensor_baseline.json
sensor_train.json
sensor_test.json
sensor_fresh.json
```

The analyst proposes generic sensor statistics, because a person must still
choose a vocabulary of possible explanations. The trained activations decide
which candidates are supported.

## Method

For each first-layer node (h_i), calculate its activation over the training
examples. For each generic candidate statistic (phi_j), calculate Pearson
correlation:

\[
\rho_{ij}=\operatorname{corr}(h_i,\phi_j)
\]

The strongest candidates across nodes become the redesigned input features.
Only a final softmax combination is trained:

\[
p(y=k|x)=\operatorname{softmax}(w_k^T[1,\phi_1(x),...,\phi_m(x)])
\]

## Inferred roles

The selected roles were:

```text
rms
mean_abs
mean_abs_change
channel_2_rms
channel_0_rms
channel_1_rms
channel_2_change
```

Examples of evidence:

- node 0 correlated with `rms` at -0.837;
- node 1 correlated with `channel_2_rms` at 0.711;
- node 3 correlated with `channel_0_rms` at 0.688.

The negative sign does not mean “bad.” It means the node activates in the
opposite direction as the candidate statistic.

## Results

| Model | Parameters | Train | Test | Fresh |
|---|---:|---:|---:|---:|
| Original MLP | 188 | 99.0% | 92.0% | 90.5% |
| Blind role reconstruction | 32 | 95.5% | 95.0% | 94.25% |

The blind reconstruction used 83% fewer parameters and improved held-out
accuracy in this run. It did not reproduce the original network's exact output;
it built a different, more constrained classifier from evidence about the
network's activations.

## What this proves

It demonstrates a real version of the original question:

> A trained model can be inspected to identify candidate computational roles,
> then replaced by a smaller model whose roles are explicit and whose final
> combination is learned.

It does not prove that each inferred statistic is the unique meaning of a node.
Correlation is evidence, not proof of causation. The next stronger test would
intervene on one candidate feature at a time, retrain with multiple seeds, and
check whether the same roles are recovered consistently.

## Reproduce

```bash
cd sensor_activity
python blind_reconstruction.py
python -m unittest test_experiment.py test_blind_reconstruction.py -v
```
