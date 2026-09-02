# Learned Minimum-Information Representation

## Objective

Find a small numeric representation directly from raw sensor windows, without providing human-designed energy, periodicity, or correlation features.

## Bottleneck result

The raw-only gated encoder was evaluated with bottleneck sizes `k = 16, 8, 4, 2, 1`. The four-dimensional bottleneck remained useful on unseen subjects:

| k | flat mean test accuracy | wide mean test accuracy |
|---:|---:|---:|
| 16 | 0.851 | 0.848 |
| 8 | 0.850 | 0.844 |
| 4 | 0.845 | 0.847 |
| 2 | 0.842 | 0.823 |
| 1 | 0.706 | 0.670 |

Under the earlier one-percentage-point tolerance, `k≈4` is a reasonable candidate capacity. This is a representation-capacity result, not yet a proof that four physical variables exist.

## Latent stability

The four latent coordinates were not compared by their raw names (`z1`, `z2`, etc.). A cross-model subspace proxy was computed instead. Its mean was `0.574` and minimum `0.391`; this is moderate rather than strong. The result does not justify treating a particular coordinate as a universal physical variable.

The next analysis therefore treats each model's latent directions as local coordinate systems and uses recurring raw-region/expression patterns as the more interpretable evidence.

## Automatic raw-to-latent mapping

For each of 10 `k=4` checkpoints, raw influence was computed separately for each latent coordinate. High-influence contiguous channel-time regions were selected algorithmically. Primitive expressions were then generated from those regions and fitted to the latent value using Lasso.

Examples of selected expressions:

```text
mean(x[6,81:97])
mean(square(x[0,71:87]))
mean(abs(diff(x[2,45:61])))
max(x[6,63:79])-min(x[6,63:79])
```

Across all 40 latent-coordinate mappings, validation R² had:

```text
mean: 0.779
SD:   0.129
range: 0.387–0.968
```

This shows that several learned latent values can be approximated by simple formulas over raw channel-time regions. It does not mean those formulas are unique or causal.

## Interpretation

The current evidence supports the following bounded conclusion:

> A raw sensor window can be compressed by the learned classifier into approximately four latent coordinates while retaining useful activity-discrimination performance. Some of those coordinates can be approximated by simple formulas over automatically selected raw channel-time regions, without predefining human feature families.

The evidence does not yet support:

```text
z1 has one fixed meaning across all models
four universal physical variables have been discovered
raw gate selection has identified the unique necessary coordinates
```

The hard-concrete gate did not yet produce a reliable sparse coordinate set. Therefore the strongest current interpretation is a learned low-dimensional representation plus candidate raw formulas, not a finalized minimum physical measurement set.

## Reproducible outputs

```text
bottleneck_gate_experiment.py
latent_symbolic_discovery.py
results/bottleneck_gate/bottleneck_gate_results.json
results/bottleneck_gate/latent_symbolic_results.json
```
