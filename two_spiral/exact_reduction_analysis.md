# Exact Mathematical Reduction Analysis

## Question

Can the trained network be made smaller while preserving exactly the same
function, rather than merely approximating it?

## Exact model function

The network computes:

```text
h1 = tanh(W1 x + b1)
h2 = tanh(W2 h1 + b2)
z  = W3 h2 + b3
p  = sigmoid(z)
```

This is already an exact mathematical representation of the model. Any shorter
exact representation must come from actual redundancy in the weights or
neurons.

## Checks performed

| Check | Result |
|---|---|
| Exact zero weights | 0 of 180 |
| Duplicate hidden neurons | none |
| Single connection removable with identical grid output | none |
| Exact duplicate-neuron merge | none |
| `W1` rank | 2 of 2 |
| `W2` rank | 12 of 12 |
| `W3` rank | 1 of 1 |

The rank of `W1` is necessarily at most 2 because the input has two dimensions.
It does not mean that ten hidden neurons can be removed: each neuron applies a
different nonlinear `tanh` after its linear projection.

## Functional removal experiment

Every individual hidden neuron was removed and compared against the original
model on the test set and a 41-by-41 input grid. The least damaging removal was
`hidden2[7]`:

- test accuracy: 86.7%, compared with the original 87.5%;
- grid decision agreement: 98.2%;
- maximum logit difference on the grid: 0.727.

This is close, but it is not exact. The output value changes for many inputs,
even when the final class often stays the same.

## Mathematical conclusion

For this particular trained model, no exact structural redundancy was found.
The 12-by-12 hidden-layer matrix is full rank, and no neurons have identical
input functions. Therefore the current evidence does not support removing a
neuron while preserving the full function for all inputs.

The distinction is important:

- **Exact function preservation:** no reduction found.
- **Same classification on sampled points:** `hidden2[7]` is nearly removable.
- **Same accuracy after retraining:** a smaller model might relearn a similar
  boundary, but that is a new model, not an exact simplification of this one.

This answers the original question honestly: the trained weights can be written
as one exact nested mathematical function, but they do not currently contain an
obvious shorter exact formula. Finding one would require discovering a hidden
coordinate transformation or symmetry in the spiral computation, not merely
deleting small numbers.

## Reproduction

```bash
cd two_spiral
python exact_reduction.py
python -m unittest test_exact_reduction.py test_bitwise.py test_polynomial.py test_spiral.py -v
```
