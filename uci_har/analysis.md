# UCI HAR Experiment Record

## Question

Can a trained model's large collection of weights guide a person to replace it
with a much smaller calculation structure while retaining its decisions on
unseen people?

## Baseline

```text
561 input features -> 64 ReLU nodes -> 32 ReLU nodes -> 6 classes
parameters: 38,246
```

The model was trained with NumPy only. The official train/test split was kept;
the test subjects were not used for training.

## Reconstruction method

1. Train the ordinary MLP.
2. Record hidden activations and class-wise responses.
3. Remove each hidden node and measure the accuracy change.
4. Rank input features by the sum of their absolute downstream weight paths.
5. Keep 128 named features from `features.txt`.
6. Standardize only those features and train a six-class softmax layer.

The selected features are explicit named calculations such as gravity
acceleration means, gyroscope correlations, entropy, autoregressive
coefficients, and frequency-domain measurements. The student does not retain
the original hidden layers.

## First result

```text
                         train       test       parameters
baseline MLP             100.00%     94.37%     38,246
selected-feature student  97.10%     93.11%        774
teacher/student agreement on test: 94.50%
```

The student retained 98.7% of the teacher's test accuracy while using about
2.0% of its parameters. This is behavioral imitation, not exact reconstruction
of the original function.

## Intervention result

The most damaging single hidden-node removals reduced test accuracy by about
1.36 percentage points. Several removals improved accuracy slightly, showing
that a large weight or active node is not automatically necessary. This supports
the workflow rule that weights are evidence of sensitivity, not node labels.

## Perturbation result

The experiment also evaluates both models after adding small input noise and
after multiplying standardized inputs by 1.1. These values are stored in
`results.json`:

```text
                         noise 0.05    input scale 1.1
baseline                   94.37%          94.37%
selected-feature student  93.04%          93.11%
```

The compact model remains close to its unmodified test result under these
small perturbations.

## Interpretation

The useful conclusion is not that each hidden node has one true human meaning.
The evidence supports a narrower claim: a trained MLP can be behaviorally
compressed into a substantially smaller model whose inputs have explicit names,
while retaining most test-set decisions.

The next stronger experiment is to use the raw inertial signals rather than the
already engineered 561 features. That will test whether human-readable
quantities such as movement energy, change, and periodicity can replace learned
temporal filters.

## Limitations

- The 561 inputs are already human-engineered features.
- The feature ranking is a sensitivity heuristic, not a causal proof.
- One random seed is not enough for a stable scientific conclusion.
- The 128-feature size is a compact baseline; a validation-based sparsity
  sweep is still needed.
