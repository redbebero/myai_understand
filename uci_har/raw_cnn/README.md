# Raw-sensor human-readable reconstruction

This is the next experiment in `human_understandable_model_workflow.md`. It
uses the UCI HAR raw inertial signals rather than the dataset's 561 engineered
features.

```bash
python3 uci_har/raw_cnn/raw_cnn_experiment.py
python3 -m unittest discover -v
```

Model:

```text
9 sensor channels × 128 time steps
→ 12 learned temporal filters, kernel size 9
→ ReLU + time average
→ 6 activity outputs
```

The analysis removes filters and sensor channels, records class-wise filter
responses, matches each filter to fixed sensor roles, tests the most important
filter pair, and compares the learned model with a role-only softmax model.

The extension also computes named local temporal roles and cross-sensor
relations, then groups them into 12 human-readable role nodes. Its comparison
is recorded in `analysis.md` and `results.json`.

The temporal relational model keeps the 8 window order instead of averaging
all windows into one value; it is the latest comparison model.

`quantization_comparison.py` measures the required control experiment: the
same CNN topology with 8-, 4-, and 2-bit weight values, alongside readable
role models.
