# Polynomial Replacement Experiment

## Goal

Test both meanings of “replace the neural network with a mathematical structure”:

1. classify the two-spiral labels directly;
2. imitate the output of the existing `2 -> 12 -> 12 -> 1` network.

## Model

Each monomial is treated as one visible calculation node:

```text
1, x, y, x², xy, y², ...
```

The output node computes:

```text
z = Σ(weight_i × monomial_i)
p = sigmoid(z)
```

The coefficients are found by ridge least squares. Degrees 1–4 are compared;
degree `d` has `(d + 1)(d + 2)/2` polynomial nodes/weights.

## Fair comparison

- Use the existing train/test spiral data.
- Keep the original neural model unchanged as the baseline.
- Fit the direct model to class labels.
- Fit the imitation model to the original model's pre-sigmoid output.
- Report accuracy, parameter count, and imitation agreement.

## Expected interpretation

A low-degree polynomial may be much smaller but fail to represent a winding
spiral boundary. That is useful evidence: the neural network may be compact in
parameters while still expressing a computation that is not compact in a plain
polynomial basis.
