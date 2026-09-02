# Human-Readable Feature-Family Compression

## Goal

The objective is not to force the model explanation down to a fixed number of individual sensor features. The objective is to add human-readable information families one at a time and measure how much of the model's `walking - walking_upstairs` score they explain on held-out validation subjects.

## Protocol

For every `flat` and `wide` checkpoint and every seed:

1. Compute the actual classifier score `score = h2 @ d_out + bias_diff`.
2. Group the raw feature dictionary into six concepts:
   - movement energy;
   - temporal periodicity;
   - temporal change;
   - acceleration–rotation coupling;
   - sensor level;
   - cross-channel coordination.
3. Fit Lasso only on fitting subjects.
4. Greedily add the family that gives the largest held-out validation R² increase.
5. Keep the test set untouched in this compression experiment.

Family-level Lasso is still allowed to choose individual features inside each selected family; the reported abstraction is the family, not one sensor statistic.

## Compression curve

Mean validation R² across five seeds:

| number of families | flat mean R² | flat SD | wide mean R² | wide SD |
|---:|---:|---:|---:|---:|
| 1 | 0.260 | 0.016 | 0.288 | 0.012 |
| 2 | 0.399 | 0.012 | 0.418 | 0.017 |
| 3 | 0.479 | 0.016 | 0.500 | 0.011 |
| 4 | 0.508 | 0.015 | 0.531 | 0.012 |
| 5 | 0.526 | 0.014 | 0.550 | 0.014 |
| 6 | 0.531 | 0.013 | 0.554 | 0.014 |

The first three families explain most of the recoverable score structure. Adding families four through six produces smaller incremental gains.

## Greedy order

The first three families were stable across all 10 model runs:

```text
temporal change
→ acceleration–rotation coupling
→ movement energy
```

The final three positions varied slightly:

```text
cross-channel coordination
↔ temporal periodicity
→ sensor level
```

This distinction matters. The exact six-family order is not universal, but the top three families were consistent across both architectures and all seeds.

## Interpretation

The compression curve supports the following human-level statement:

> The model's walking-versus-upstairs score is primarily represented by temporal change, acceleration–rotation coordination, and movement energy. Temporal periodicity, cross-channel coordination, and sensor level add smaller but measurable information.

The result is not that one of the first three families alone is sufficient. The mean R² rises from approximately 0.26–0.29 with one family to approximately 0.48–0.50 with three families, then to approximately 0.53–0.55 with all six.

In plain language:

```text
one movement concept        → partial explanation
three concepts              → most of the recoverable structure
six concepts                → modest additional detail
```

## Caveat about greedy validation selection

The family order is selected by validation R², so the displayed validation curve is an explanatory selection curve and can be optimistic. It is not a final independent test estimate. The test set remains unused here and should be used only for a preregistered final family set or final intervention evaluation.

## Final conclusion

The useful compression unit is the information family, not the individual feature. Across `flat` and `wide` models, a compact human-readable description is:

> Changes in the movement signal, coordination between acceleration and body rotation, and overall movement energy account for most of the model's walking-versus-upstairs score structure. Periodicity, cross-channel coordination, and sensor level refine that explanation rather than replacing it.
