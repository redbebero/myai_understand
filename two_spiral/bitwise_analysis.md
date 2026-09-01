# Bitwise Weight Simplification

## Experiment

The original network uses real-valued weights, `tanh`, and `sigmoid`. This
experiment changes only the representation of its learned parameters:

- **8-bit, 4-bit, 2-bit:** round each parameter to a small signed grid;
- **binary weights:** replace each weight by its sign and retain a layer scale;
- **binary network:** convert inputs and hidden activations to `-1/+1` and
  calculate each dot product with the equivalent XNOR-plus-popcount rule.

For a binary vector, the dot product is:

\[
 a\cdot b=2\,\mathrm{popcount}(\mathrm{XNOR}(a,b))-n
\]

The code writes the measured results to `bitwise_results.json`.

## Results

| Variant | Train accuracy | Test accuracy |
|---|---:|---:|
| Original real weights | 94.2% | 87.5% |
| 8-bit weights | 94.2% | 87.5% |
| 4-bit weights | 89.2% | 84.2% |
| Binary weights (`±1`) | 50.8% | 51.7% |
| Binary network | 55.0% | 55.8% |

The 8-bit model produced the same measured accuracy as the original model.
The 4-bit model lost only 3.3 percentage points on the test set. In contrast,
keeping only the sign of each weight reduced performance to approximately
chance level. Binarizing the inputs and hidden activations did not recover the
lost information.

## Interpretation

This is a representation experiment, not a new training method. The network's
topology stays the same, so it answers:

> How much of the learned behavior is preserved when real-valued weights are
> replaced by a small number of discrete values or signs?

If 8-bit accuracy stays close to the original, most weight precision was
redundant. If binary-network accuracy falls sharply, the learned decision rule
depends on magnitude and smooth activations, not only connection signs.

That is what happened here. The important information is not just whether a
connection is positive or negative. The relative magnitudes of the weights and
the continuous `tanh` activations shape the boundary. However, much of the
precision below roughly four bits was unnecessary for this trained model.

This gives a more precise conclusion than “bit operations can replace the
model”: the model is tolerant of low-precision integer representation, but its
computation is not reducible to signs and repeated XNOR operations without
retraining or redesigning the architecture.

The result should not be described as “the weights became a repeated bit
algorithm.” The valid conclusion is narrower: the learned calculation can, or
cannot, tolerate a lower-precision representation. A genuinely hand-designed
bit algorithm would require replacing the learned topology and selecting its
rules without fitting the original weights.

## Reproduce

```bash
cd two_spiral
python bitwise_experiment.py
python -m unittest test_bitwise.py test_polynomial.py test_spiral.py -v
```
