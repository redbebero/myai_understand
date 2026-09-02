# Hard-Concrete/L0 Gate and Bottleneck Sweep

## Executed protocol

The raw input was passed directly to a gated encoder:

```text
raw 9×128
→ hard-concrete-style gate
→ hidden 64/96
→ hidden 16
→ bottleneck k=4
→ 6-class classifier
```

The sweep used `flat` and `wide` encoders, five seeds, and L0 penalties `0`, `0.03`, `0.1`, and `0.3`. The test set was evaluated only after training.

## k≈4 result

With a four-dimensional bottleneck, unseen-test accuracy remained close to the raw MLP reference:

| architecture | mean test accuracy |
|---|---:|
| flat | 0.867 |
| wide | 0.873 |

The raw MLP references were approximately `0.841` (flat) and `0.849` (wide). This confirms that a learned `k=4` bottleneck can retain useful class information in this implementation.

## Gate result

The gate regularizer did not produce a meaningful coordinate-level sparsity curve. For nonzero penalties, the relaxed gate values collapsed close to the lower stretch boundary for essentially every coordinate, while the hard threshold count was zero for all runs. The reported expected-active count was not consistent with the deterministic gate threshold because the expected-probability formula belongs to the stochastic hard-concrete distribution, whereas this first implementation used a deterministic relaxation.

Therefore the current gate results must **not** be interpreted as:

```text
only N raw coordinates are sufficient
```

The gate implementation is a useful diagnostic, but it needs the actual stochastic hard-concrete sampling/straight-through estimator, or a consistent deterministic L0 surrogate, before active raw channel-time positions can be trusted.

## What is established

- The raw-only bottleneck experiment supports approximately four latent coordinates as a useful candidate capacity.
- The model can be trained without human-defined sensor features.
- The current L0 gate objective exposes a parameterization/regularization problem rather than a reliable raw-coordinate selection.

## Next correction

Implement the stochastic hard-concrete gate with:

```text
u ~ Uniform(0,1)
s = sigmoid((log(u)-log(1-u)+log_alpha)/temperature)
stretched = s*(zeta-gamma)+gamma
gate = clip(stretched, 0, 1)
```

Use the expected nonzero probability only for the L0 penalty and use sampled gates in the forward pass. Then rerun the lambda sweep and only afterward aggregate recurrent channel-time coordinates and trace the four latent coordinates.
