# Two-Spiral Model Analysis

## Question

Can a model learn a pattern that is difficult to express with simple rules, and can its essential computation be identified by removing parts of the model?

## Baseline

Architecture: 2 inputs → 12 hidden neurons → 12 hidden neurons → 1 output
Training examples: 240
Test examples: 240
Training accuracy: 97.1%
Test accuracy: 91.7%
Nonzero weights: 180

## Computation samples

The full intermediate activations are saved in `spiral_analysis.json`. They show how each input becomes hidden-layer values and then an output probability.

## Removal results

Least damaging single connection removal: `hidden1→hidden2[2,1]`; accuracy 92.5%.
Most damaging single neuron removal: `hidden1[9]`; accuracy 50.0%.
Test accuracy after pruning that neuron and retraining: 92.5%.

Repeated training results are saved in `spiral_analysis.json` for seeds 11, 23, and 41.

## Interpretation

The two-spiral task is larger than XOR and includes unseen test points. A component that can be removed with little accuracy loss may be redundant for this trained model. A component that causes a large loss contributes to the current computation, but is not automatically necessary in every possible model.

The model recovered most of its baseline test performance after pruning `hidden1[4]` and retraining: 85.8% versus the original 87.5%. This suggests the removed neuron was important to the original calculation, but not structurally indispensable.

The repeated short training runs produced lower scores than the baseline. This shows that initialization and training duration affect the result, so one trained model cannot establish a general minimum structure.

## Limitations

This experiment uses one architecture, one data-generation process, and one fully trained baseline seed. The repeated seeds use shorter training to check sensitivity, not to claim equal optimization. Repeat with other architectures and noise levels before claiming a general rule.
