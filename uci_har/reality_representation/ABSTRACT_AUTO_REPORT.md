# Abstracted AI representation

## Objective

Raise the automatic formulas to a human-readable concept level rather than exposing every raw expression separately.

## Method

For each bottleneck checkpoint, all AI-discovered raw regions were expanded into primitive expressions. Expressions were grouped by mathematical operation:

- `movement_energy`: local mean of squared signal
- `temporal_change`: local mean absolute first difference
- `local_range`: max minus min
- `trend`: local slope
- `sensor_coordination`: cross-sensor product or correlation
- `sensor_level`: local mean

Within each group, every expression was standardized using the fitting subjects and averaged. The result was one concept value per group. A six-class linear softmax classifier used these concept values; the official test subjects were not used to construct the concepts or fit the classifier.

The resulting representation had six operation-level concepts instead of 273–312 individual candidate expressions.

## Results

Mean across ten flat/wide, five-seed runs:

| representation | mean test accuracy |
|---|---:|
| one discovered expression | 0.3615 |
| six operation-level concepts | 0.6628 |
| previous automatic expressions | 0.7432 |
| expanded automatic expressions | 0.8039 |
| human top three families | 0.8799 |

The abstracted representation ranged from **0.6105 to 0.6922** across runs.

## Interpretation

The operation-level concepts are substantially more informative than a single expression, but naïvely averaging every expression in a group discards too much information. The experiment demonstrates that a human-readable abstraction layer is feasible, but the abstraction must preserve important time-window, sensor-axis, and latent-specific structure.

The next abstraction should use weighted or attention-like aggregation within each concept and retain a small number of stable subcomponents, for example separate axis-specific movement energy and rotation energy, instead of averaging all expressions indiscriminately.
