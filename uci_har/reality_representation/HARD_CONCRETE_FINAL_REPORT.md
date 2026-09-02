# Raw-Only Bottleneck and Hard-Concrete Gate Outcome

## What was executed

A raw-only model was trained with:

```text
9×128 raw sensor values
→ stochastic hard-concrete-style gate
→ hidden layers
→ k=4 bottleneck
→ activity classifier
```

The gate objective added an L0-style expected nonzero penalty. Four penalty values (`0`, `0.03`, `0.1`, `0.3`) were evaluated for both `flat` and `wide` networks across five seeds.

## Result

The four-dimensional bottleneck remained accurate:

```text
flat mean test accuracy:  0.890
wide mean test accuracy:  0.889
```

The raw MLP references were approximately `0.841` (flat) and `0.849` (wide), although this comparison uses a different training implementation and should be treated as an engineering comparison rather than a controlled benchmark.

The gate, however, did not produce a reliable sparse subset. Across nonzero penalties:

```text
expected active count: about 946 / 1152
hard probability count: 1152 / 1152
```

The relaxed gate values were pushed toward a low value, but the expected-probability threshold still counted every coordinate as active. This means the current parameterization does not provide a trustworthy active-coordinate curve.

## Consequence for the planned sequence

The following stage is valid:

```text
learned bottleneck capacity
→ k≈4 remains a useful candidate
```

The following stages are not yet valid:

```text
recurrent surviving raw coordinates
→ latent z1–z4 raw attribution
→ final human interpretation
```

There is no defensible sparse survivor set to aggregate yet.

## Required correction

Before extracting surviving channel-time positions, replace this implementation with one of:

1. a verified hard-concrete implementation whose sampled gate, expected L0 probability, and evaluation gate use the same temperature/stretch convention; or
2. a group-L0 gate over channel-time blocks with an explicit proximal/threshold step, followed by a validation sweep.

The current results are saved for reproducibility but are intentionally not presented as evidence that a small number of raw coordinates is sufficient.
