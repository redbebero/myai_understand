# Advanced Input-Trace Report

## Scope

This is the follow-up analysis for `walking` versus `walking_upstairs`. It evaluates both MLP widths, uses a real L1-penalized regression, aggregates feature families, applies family-specific interventions, and chooses controls from the same family with similar baseline influence.

## Model coverage

Ten models were evaluated:

- `flat`: `1152 → 64 → 32 → 6`, five seeds.
- `wide`: `1152 → 96 → 32 → 6`, five seeds.

Mean test accuracy:

| model | mean | range |
|---|---:|---:|
| flat | 0.8410 | 0.8286–0.8527 |
| wide | 0.8487 | 0.8395–0.8582 |

The wide model is slightly more accurate on average, but both architectures were analyzed using the same raw-input trace protocol.

## True sparse regression

The score explanation uses coordinate-descent Lasso over the sensor feature dictionary. The selected coefficient is the largest-magnitude nonzero coefficient, not simply the first feature in dictionary order.

Observed nonzero counts were:

- flat: 56–83 features;
- wide: 45–60 features.

Therefore Lasso was implemented correctly, but this dataset/model combination did **not** reduce the explanation to 4–6 features. The result is sparse relative to the full dictionary, but not highly sparse. A stronger sparsity penalty or a preregistered one-standard-error selection rule would be needed to force a smaller dictionary, with an explicit risk of lower validation R².

Held-out validation R² ranged from 0.4592 to 0.6389 for flat and 0.5247 to 0.8014 for wide. The wide architecture produced the stronger score approximation in most seeds.

## Feature-family recurrence

Family recurrence counts a family once per model if at least one Lasso coefficient from that family is nonzero.

| family | recurrence |
|---|---:|
| movement energy | 10/10 |
| sensor level | 10/10 |
| temporal change | 10/10 |
| temporal periodicity | 10/10 |
| acceleration–rotation coupling | 10/10 |
| cross-channel correlation | 10/10 |
| peak | 10/10 |
| peak-to-peak | 4/10 |

The first seven families are broad and all recur because the selected Lasso solutions remain moderately dense. This is evidence of recurring information classes, not evidence that every family is independently necessary.

## Influence-map stability

For each model, the mean absolute `9 × 128` input-influence map was normalized and compared with pairwise cosine similarity across all ten models.

```text
mean cosine similarity: 0.9265
minimum:                 0.9067
maximum:                 0.9403
```

This is stronger architecture/seed stability than the exact top feature names. It suggests the models attend to similar raw-sensor regions in aggregate, while exact feature decomposition remains non-unique.

## Family-specific interventions

Interventions were applied to the channel associated with the strongest Lasso coefficient:

- energy: preserve the mean and reduce centered amplitude to 80%;
- level: shift the channel mean while preserving within-window changes;
- temporal periodicity: permute the time order, preserving the sample values but disrupting temporal order;
- coupling/correlation and other temporal features: circularly shift the channel, preserving its marginal distribution while disrupting alignment.

The matched control was selected from the same feature family, on a different channel, minimizing the difference in baseline absolute input influence from the selected channel.

The resulting intervention is more controlled than the earlier arbitrary adjacent-channel shift. However, it still changes multiple correlated properties. For example, permuting time destroys more than one possible periodicity statistic, and reducing energy can alter nonlinear downstream responses.

## Interpretation

Across both architectures and five seeds, the strongest stable statement is:

> The classifier uses distributed information about movement magnitude, sensor level, temporal organization, and coordination between acceleration and rotation signals. The exact minimal feature set is not stable; the aggregate raw-input influence pattern is much more stable.

The Lasso results do not justify saying that only four or five features are sufficient. Instead, they show that a penalized linear approximation of the nonlinear score repeatedly retains several human-readable feature families.

## Causal boundary

The intervention/control results are diagnostic, not causal proof. A causal claim would require stronger nuisance controls, matched perturbation magnitudes in feature space, repeated perturbations, and uncertainty intervals. The current evidence supports:

```text
association with the model score
+ stable input influence pattern
+ family-specific perturbation sensitivity
```

It does not establish:

```text
one sensor or feature family is physically necessary
```

## Final conclusion

The reverse-tracing pipeline now covers both model widths, true Lasso explanations, automatic family recurrence, influence-map stability, feature-specific interventions, and same-family matched controls. The resulting human-level explanation is:

> Walking versus walking upstairs is represented as a distributed movement pattern: how much the body accelerates, how the signal changes and repeats through the time window, and how acceleration aligns with rotational motion. These information families recur across flat and wide networks, while the exact feature selected as “most important” varies and should not be promoted to a unique mechanism.
