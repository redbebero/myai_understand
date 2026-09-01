# Larger Weight-Guided Sensor Redesign

## Scale-up

The original sensor experiment used 3 channels and 12 time steps. This repeat
uses:

```text
6 channels × 24 time steps = 144 inputs
16 first-layer nodes
12 second-layer nodes
4 output classes
```

The data has 100 examples per class, random phase and amplitude variation,
sensor noise, and 5% label noise. The larger baseline has 2,576 trainable
parameters.

## Weight-guided analysis

The model was trained first. It was then inspected without using the generator
inside the role-analysis step. Each hidden activation was compared with generic
time-series candidates such as overall RMS, mean change, and channel-specific
RMS/change statistics.

The strongest recurring candidate was `mean_change`. Several nodes also showed
large relationships with the changes in channels 3, 4, and 5. A different group
showed strong relationships with overall RMS and channel-specific RMS.

The selected roles were:

```text
mean_change
channel_4_change
channel_5_change
channel_3_change
rms
mean_abs
channel_4_rms
channel_5_rms
channel_2_rms
channel_3_rms
```

These roles are not claimed to be the literal meanings of individual neurons.
They are human-readable candidate calculations supported by activation
correlations with the learned model.

## Results

| Model | Inputs / role features | Parameters | Train | Test | Fresh |
|---|---:|---:|---:|---:|---:|
| Larger neural baseline | 144 | 2,576 | 100.0% | 95.5% | 95.0% |
| Blind role redesign | 10 fixed roles | 44 | 92.75% | 91.25% | 93.25% |

The redesign uses about 98.3% fewer trainable parameters. It loses test
accuracy in this run but performs well on fresh data. This is an important
contrast with the smaller experiment: a role vocabulary that worked at small
scale does not automatically preserve a larger model's computation.

## Interpretation for the research question

The larger experiment strengthens the question rather than producing a simple
success story. Weight and activation analysis can identify recurring candidate
roles even when the model is larger, but selecting those roles by correlation is
not enough to reproduce the full learned function. The larger model combines
more time points and channels in ways that are distributed across its neurons.

The current evidence supports this conclusion:

> A trained model's weights can guide a human-readable redesign, but the number
> of roles, their interactions, and the amount of retraining needed grow with
> model complexity. A small redesign may generalize well while still failing to
> reproduce the original model exactly.

## Reproduction

```bash
cd sensor_activity
python large_experiment.py
python -m unittest test_large_experiment.py -v
```
