# Raw-Only Bottleneck and Gate Experiment

## Goal

Learn a compact representation directly from the `9 × 128` raw sensor values without supplying energy, periodicity, correlation, or other human-designed features.

The model is:

```text
raw 1152
→ learnable raw gate
→ hidden 64 or 96
→ hidden 16
→ bottleneck k
→ 6-class classifier
```

The training objective is cross-entropy plus a gate-size penalty. Five seeds were run for each `k` and both width variants.

## Unseen-test accuracy

| bottleneck k | flat mean accuracy | wide mean accuracy |
|---:|---:|---:|
| 16 | 0.851 | 0.848 |
| 8 | 0.850 | 0.844 |
| 4 | 0.845 | 0.847 |
| 2 | 0.842 | 0.823 |
| 1 | 0.706 | 0.670 |

Raw-input MLP references were approximately `0.841` for flat and `0.849` for wide.

## Interpretation

The bottleneck sweep gives direct evidence about latent information capacity:

- `k = 16`, `8`, and `4` preserve accuracy close to the raw MLP reference;
- `k = 2` is architecture-dependent: flat remains close, wide drops noticeably;
- `k = 1` loses substantial discriminative information.

A bounded conclusion is therefore:

> For this split and training setup, the classifier can compress the raw signal to roughly 4 latent coordinates without a large average accuracy loss. One latent coordinate is insufficient, and two coordinates are not reliably sufficient across architectures.

This is a learned bottleneck result; no semantic feature candidates were supplied.

## Gate result and limitation

The learnable gates did not become clean binary selectors under the current sigmoid parameterization. At threshold `g > 0.5`, essentially all raw coordinates remained active. The mean gate value was approximately `0.71`.

Therefore this run successfully tests latent bottleneck size, but it does **not** yet establish that a small percentage of the 1,152 raw coordinates is sufficient. The gate needs a hard-concrete or another explicit L0 relaxation before making a claim about raw-coordinate sparsity.

## Next technically correct gate step

Replace the sigmoid gate with a hard-concrete / stretched-concrete gate and penalize the expected number of nonzero gates. Then sweep the L0 penalty and report:

```text
active raw-coordinate count
vs.
unseen-test accuracy
```

The current report intentionally does not claim that the gate selected a sparse raw subset.
