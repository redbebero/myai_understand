# UCI HAR: Weight-Guided Human-Readable Reconstruction

This experiment follows `human_understandable_model_workflow.md` on the UCI
Human Activity Recognition Using Smartphones dataset.

Run:

```bash
python3 uci_har/uci_har_experiment.py
python3 -m unittest uci_har.test_uci_har -v
```

The script downloads the official UCI archive on the first run, trains a
`561 -> 64 -> 32 -> 6` ReLU MLP, records activations, removes hidden nodes,
and ranks input features by the learned downstream weight path. It then trains
only a final softmax layer on the top 128 named UCI features as a compact,
human-readable student model.

The 128-feature choice is fixed before test evaluation in this first pass. It
is a compact reconstruction baseline, not a claim that the selected features
are the unique meaning of the original nodes.
# UCI HAR experiments

The original weight-guided reconstruction is recorded in `analysis.md`.

The interaction follow-up measures single-node and pair ablations, ranks
`D(i,j)-D(i)-D(j)`, records conditional joint activation and class changes,
and compares a compact named-feature student model. Reproduce it with:

```bash
python -m uci_har.interaction_experiment
python -m unittest uci_har.test_uci_har uci_har.test_interaction_experiment -v
```

Results are written to `interaction_results.json` and interpreted in
`interaction_analysis.md`.
