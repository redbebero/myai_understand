# Sensor Activity: Weight-Guided Redesign

This experiment studies a problem whose useful decision rule is not supplied to
the redesign stage: noisy multivariate sensor sequences classified as idle,
walk, run, or turn.

Run the complete experiment with:

```bash
python experiment.py
python -m unittest test_experiment.py -v
```

`experiment.py` trains a small baseline MLP, removes each first-layer node,
records the accuracy drop, and builds a human-readable redesign from fixed
sensor-role features. Only the final softmax weights of the redesign are
learned.
