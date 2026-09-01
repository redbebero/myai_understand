# Dynamic Sparse Experiment

## Question

Can a small AI learn while removing weak connections and growing new connections, instead of keeping a fully connected structure throughout training?

## Model

Two-spiral classification:

```text
2 inputs → 12 hidden neurons → 12 hidden neurons → 1 output
```

The dense model has 180 possible weights. This experiment keeps 50% active: 90 connections. Every 100 epochs, one active connection per layer is removed and one inactive connection with the largest accumulated gradient is grown.

## Comparison

Compare this result with the dense model in `../two_spiral/` using:

- Test accuracy
- Active connection count
- Rewiring history
- Accuracy after each rewiring event

## Interpretation rule

Do not conclude that dynamic connections are better from one run. Repeat with other seeds and compare equal training budgets.
