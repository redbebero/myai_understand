# Input-Trace Analysis Report

## Question

For `walking` versus `walking_upstairs`, which raw sensor information contributes to the model's actual two-class decision, and how can that information be expressed in human-readable terms?

## Executed method

The trained model used all `9 × 128 = 1,152` raw sensor values. For each sample, the output direction was defined from the full classifier weight matrix:

```text
d_out = W3[:, walking] - W3[:, walking_upstairs]
score = h2 @ d_out + (b3[walking] - b3[walking_upstairs])
```

The local input influence used the sample's ReLU masks:

```text
v = W1 D1 W2 D2 d_out
```

where `D1` and `D2` select active ReLU units. Each `v` was reshaped to `9 × 128` and saved for class-conditional signed and absolute influence maps.

The exact trained parameters were saved for every seed under:

```text
results/input_trace/model_seed_7.npz
results/input_trace/model_seed_11.npz
results/input_trace/model_seed_19.npz
results/input_trace/model_seed_23.npz
results/input_trace/model_seed_31.npz
```

## Direct score explanations

The two-class score was explained directly, not through an individual hidden neuron or hidden-axis label. Feature selection used the fitting subjects; validation `R²` was then measured on held-out validation subjects.

| seed | test accuracy | top input-influence channel | top direct-score feature | validation R² |
|---:|---:|---|---|---:|
| 7 | 0.8307 | `total_acc_x` | `total_acc_y:energy` | 0.5105 |
| 11 | 0.8463 | `total_acc_y` | `body_gyro_z×total_acc_x:correlation` | 0.2744 |
| 19 | 0.8527 | `body_acc_z` | `total_acc_z:autocorrelation_lag1` | 0.2501 |
| 23 | 0.8286 | `total_acc_x` | `body_gyro_z:energy` | 0.2478 |
| 31 | 0.8466 | `total_acc_x` | `total_acc_z:energy` | 0.3186 |

The top exact feature was not identical across seeds. Recurrent families were:

- acceleration energy;
- sensor level;
- temporal autocorrelation / periodicity;
- acceleration–gyroscope or cross-axis correlation.

This supports family-level interpretation more strongly than any single feature name.

## Human-readable interpretation

The most defensible abstraction for this model pair is:

> The model distinguishes walking from walking upstairs using a combination of movement magnitude, how acceleration changes over time, and how acceleration and rotation signals are coordinated. No single raw sensor value was stable as the explanation across all seeds.

A more concrete seed-7 description is:

> In seed 7, the two-class score was most strongly associated with total-acceleration energy and level-related features, while the largest absolute input influence was concentrated in `total_acc_x`. This indicates that the model used the amount and distribution of acceleration over the window, not an isolated reading.

A concrete seed-11 description is:

> In seed 11, the top direct score feature was the zero-lag correlation between `body_gyro_z` and `total_acc_x`. This is consistent with a candidate interpretation in terms of acceleration–rotation coordination, but it is not stable enough across seeds to be called a universal mechanism.

## Intervention result

The top selected feature was mapped to a channel and tested by a smooth circular time shift. A neighboring channel received the same type of shift as a matched control.

| seed | selected feature | intervention accuracy drop | matched-control accuracy drop |
|---:|---|---:|---:|
| 7 | `total_acc_y:energy` | -0.0014 | 0.0024 |
| 11 | `body_gyro_z×total_acc_x:correlation` | 0.0546 | 0.0143 |
| 19 | `total_acc_z:autocorrelation_lag1` | 0.0007 | 0.0730 |
| 23 | `body_gyro_z:energy` | 0.0611 | 0.0129 |
| 31 | `total_acc_z:energy` | 0.0037 | 0.0763 |

The intervention effect was not consistently larger than the control. Therefore the feature families are computational explanations, not established causal sensor mechanisms.

## What this experiment establishes

- The actual two-class classifier direction can be traced through both ReLU layers to all 1,152 raw inputs.
- The resulting input influence has a channel-by-time structure and is saved by seed.
- The direct score can be statistically explained by sensor feature families on held-out validation data.
- Exact top features vary across seeds, while broader families recur.
- Human-readable descriptions can be made at the level of movement magnitude, temporal autocorrelation, and acceleration–rotation coordination.

## What it does not establish

- One raw sensor value is universally necessary.
- One feature family is the physical cause of the prediction.
- A feature's correlation with the score proves causality.
- A timestep or latent direction number has the same meaning across seeds.
- The current control intervention selectively isolates one physical relationship.

## Final bounded conclusion

```text
raw sensor values
→ full two-class model score
→ samplewise ReLU-gated input influence
→ sensor-channel and time pattern
→ direct feature-family explanation
→ human-readable movement abstraction
```

For this dataset and model, the safe conclusion is:

> The walking-versus-upstairs decision is associated with distributed raw-sensor information involving acceleration magnitude, temporal regularity, and cross-sensor coordination. The analysis can trace this information from the actual classifier direction back to the input and summarize it in human-readable terms, but the present intervention controls do not justify calling any one sensor value or relation a proven cause.
