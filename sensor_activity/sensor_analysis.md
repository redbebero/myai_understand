# Sensor Activity Model Analysis

## Question

Can a model learn a noisy sensor-activity classification task, and can its
learned nodes be analyzed and replaced by a small set of human-readable
calculation roles?

## Task

Each example contains 3 sensor channels over 12 time steps:

```text
36 real-valued measurements -> idle / walk / run / turn
```

The data contains Gaussian sensor noise, random phase/amplitude variation, and
5% label noise. The generation code is used to create held-out data, but the
redesign does not copy its class formulas.

## Baseline model

```text
36 inputs -> 4 tanh nodes -> 4 tanh nodes -> 4-class softmax
```

It has 188 trainable scalar parameters. The baseline results are:

| Split | Accuracy |
|---|---:|
| Training | 99.0% |
| Held-out test | 92.0% |
| Fresh generated data | 90.5% |

## Weight-only node analysis

Each first-layer node was removed while leaving the rest of the original model
unchanged. The largest accuracy drops were:

| Node | Accuracy after removal | Drop |
|---:|---:|---:|
| 3 | 67.5% | 31.5 percentage points |
| 0 | 74.0% | 25.0 percentage points |
| 1 | 74.75% | 24.25 percentage points |
| 2 | 78.5% | 20.5 percentage points |

This is evidence from the learned model. It does not claim that node 3 has a
human semantic meaning by itself; it says the current trained calculation
depends strongly on it.

## Human-readable redesign

After observing the task's sensor dimensions, I chose generic role features that
can be explained without referring to the hidden class-generation equations:

```text
mean_abs             overall movement magnitude
rms                  signal energy
mean_step_change     average frame-to-frame change
alternating_change   rapid change in the direction of change
channel_0_mean       average level of channel 0
channel_1_mean       average level of channel 1
channel_2_mean       average level of channel 2
```

The model is:

\[
\phi(x)=[1,\text{meanAbs},\text{RMS},\text{stepChange},
\text{alternation},\text{channel means}]
\]

\[
p(y=k|x)=\operatorname{softmax}(w_k^T\phi(x))
\]

The role equations are fixed by hand. Only the 32 final softmax coefficients
are learned.

| Model | Parameters | Train | Test | Fresh |
|---|---:|---:|---:|---:|
| Baseline MLP | 188 | 99.0% | 92.0% | 90.5% |
| Role-based redesign | 32 | 93.5% | 93.0% | 92.25% |

## Interpretation

The redesign used 83% fewer parameters and slightly improved held-out and fresh
accuracy in this run, although it did not fit the noisy training set as well.
That suggests the fixed roles acted as a useful inductive bias and reduced the
baseline's ability to memorize label noise.

The result is not proof that the redesign recovered the exact internal meaning
of the MLP's nodes. The role features were designed after inspecting the sensor
problem, while the node-importance ranking came from the trained weights. The
experiment therefore separates two claims:

1. **Weight evidence:** ablation identifies which learned nodes the original
   model currently relies on.
2. **Human redesign:** interpretable sensor roles can replace a large learned
   representation and retain the decision task with fewer parameters.

The important failure/limitation is that node importance alone does not reveal a
unique semantic label. A node can combine several sensor properties. To make the
next experiment stronger, hide the data-generation code from the analyst, use
multiple random seeds, and infer the role features only from activation-response
correlations and intervention tests.
