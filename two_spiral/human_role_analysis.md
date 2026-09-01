# Human-Designed Node-Group Model

## Purpose

This model is not trained. It is a direct human design made from named node
groups:

```text
(x, y)
  ├─ distance nodes: radius and center confidence
  ├─ direction nodes: signs of x/y and angle
  └─ spiral nodes: phase agreement with a rotating arm
                  ↓
              hand-written vote
                  ↓
             class 0 or 1
```

The central hypothesis is:

```text
radius = sqrt(x² + y²)
angle = atan2(y, x)
phase = angle - 4π × radius
```

The two spiral classes are approximately opposite phases. Therefore the model
uses `cos(phase)` as the arm detector and votes for class 1 when the opposite
arm has the larger response. The constant `4π` and all combination rules are
written by hand; no labels are used to fit them.

## Results

| Model | Train | Existing test | Fresh generated data | Nodes |
|---|---:|---:|---:|---:|
| Existing trained neural network | 94.2% | 87.5% | 88.3% | 24 hidden nodes |
| Human full model | 95.8% | 97.5% | 97.5% | 9 named nodes |
| Distance only | 50.0% | 50.0% | 50.0% | 9 named nodes |
| Direction only | 50.0% | 50.0% | 50.0% | 9 named nodes |
| Spiral only | 95.8% | 97.5% | 97.5% | 9 named nodes |
| Distance + spiral | 95.8% | 97.5% | 97.5% | 9 named nodes |

## Interpretation

The spiral phase group is the decisive group for this dataset. Distance alone
cannot distinguish the classes because both classes use the same range of
radii. Direction alone also cannot distinguish them because both arms pass
through the same directions. The phase combines angle and radius, which is the
geometric relationship that separates the two generated arms.

This is a meaningful result, but it must be described accurately. The model did
not discover `4π` from the neural weights. I inspected the task geometry and
designed a hypothesis. Therefore this experiment answers:

> Can a person design interpretable node roles that solve the task without
> gradient-based learning?

It does not yet answer whether the trained neural network's weights themselves
contain this exact `4π` formula. That is a separate, harder comparison.

The important structural result is that 9 named calculations were enough for
this synthetic task, while the trained model used two hidden layers of 12
neurons. The smaller model is not automatically more general: it succeeds here
because its designer used the spiral's geometric rule directly.

## Why this matters for the original question

The experiment separates three things that were previously mixed together:

1. **Learning:** adjust unknown weights from examples.
2. **Reading a model:** inspect learned weights and explain their effects.
3. **Human design:** choose node roles and constants from a structural
   hypothesis.

The present model demonstrates the third case. The next rigorous comparison is
to hide the data-generation formula, inspect only the trained weights, and see
whether the same role groups can be reconstructed without using `4π` directly.
