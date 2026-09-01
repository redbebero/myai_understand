# Polynomial Model Analysis

## What was built

`polynomial_experiment.py` creates two models:

- **Direct polynomial classifier:** coefficients fit to the spiral labels.
- **Neural imitation polynomial:** coefficients fit to the original network's
  output before its final sigmoid.

The polynomial nodes are explicit monomials. For degree 4 there are 15 nodes:

```text
1, x, y, x², xy, y², x³, x²y, xy², y³,
x⁴, x³y, x²y², xy³, y⁴
```

The final calculation is:

```text
z = w₀ + w₁x + w₂y + w₃x² + ... + w₁₄y⁴
p = 1 / (1 + exp(-z))
```

## Results

The current baseline network scores **87.5%** on the held-out test set.

| Model | Degree | Parameters | Test accuracy | Neural agreement |
|---|---:|---:|---:|---:|
| Direct polynomial | 1 | 3 | 53.3% | — |
| Direct polynomial | 2 | 6 | 55.0% | — |
| Direct polynomial | 3 | 10 | 57.5% | — |
| Direct polynomial | 4 | 15 | 63.3% | — |
| Imitation polynomial | 1 | 3 | 55.8% | 53.3% |
| Imitation polynomial | 2 | 6 | 54.2% | 53.3% |
| Imitation polynomial | 3 | 10 | 51.7% | 55.8% |
| Imitation polynomial | 4 | 15 | 56.7% | 54.2% |

## Interpretation

The polynomial model is much smaller, but degree 4 is still far below the
neural baseline. This gives a concrete answer:

> A plain low-degree polynomial of `(x, y)` does not preserve the spiral
> decision rule or the trained network's behavior on this dataset.

The likely reason is structural. A polynomial in Cartesian coordinates makes a
smooth algebraic boundary. The spiral boundary winds around the origin, so a
small Cartesian polynomial cannot create enough turns without a much higher
degree and many more coefficients.

The important next question is not “try random higher degrees.” It is:

> Which coordinate transformation makes the network's computation simpler?

The natural next candidate is polar structure:

```text
r = sqrt(x² + y²)
theta = atan2(y, x)
```

Then test a small equation involving `theta` and `r`, such as a threshold on
`theta - k*r`. This is a hypothesis based on the geometry of the data, not blind
coefficient search.

## Reproduction

```bash
cd two_spiral
python polynomial_experiment.py
python -m unittest test_polynomial.py test_spiral.py -v
```
