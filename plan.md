# Hidden Representation Compression Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether a frozen classifier preserves its decisions when hidden representations are compressed by random, variance-preserving, or class-separating directions.

**Architecture:** Reuse the existing UCI HAR `561→64→32→6` MLP and its trained parameters. Fit every compression basis on train hidden representations only, reconstruct compressed vectors in the original 32-dimensional space, and pass them through the frozen output layer; validation is diagnostic and test is held out for final comparison.

**Tech Stack:** Python 3, NumPy, stdlib `unittest`, existing repository helpers.

**Spec:** User-approved representation-compression design in the conversation.

## Global Constraints

- Do not retrain the compressed representations or output layer.
- Use the existing strict train/validation/test split.
- Evaluate `k=(32,16,8,4,2,1)`.
- Fit PCA and supervised directions on train representations only.
- Report accuracy, prediction agreement, loss, reconstruction error, class separation, and distance-relation correlation.
- Use five model seeds and twenty random projections per model seed.
- Add no dependencies.

---

### Task 1: Lock the projection behavior with tests

**Files:**
- Create: `uci_har/test_representation_compression_experiment.py`
- Create: `uci_har/representation_compression_experiment.py`

**Interfaces:**
- `project_and_reconstruct(representations, basis, mean)` returns reconstructed representations with the original shape.
- `pairwise_distance_correlation(first, second)` returns a finite scalar in `[-1, 1]`.
- `class_separating_basis(representations, labels, k)` returns an orthonormal basis with `k` columns.

- [x] Write tests for reconstruction shape, orthonormal class basis, and perfect distance correlation.
- [x] Run `python -m unittest uci_har.test_representation_compression_experiment -v`; expect import failure before implementation.
- [x] Implement the three minimal helpers with NumPy only.
- [x] Re-run the focused tests and require PASS.

### Task 2: Implement frozen-model compression evaluation

**Files:**
- Modify: `uci_har/representation_compression_experiment.py`
- Modify: `uci_har/test_representation_compression_experiment.py`

**Interfaces:**
- `_evaluate_method(model, train_h, eval_h, labels, basis, mean)` returns metrics without changing `model`.
- `run_representation_compression(data_dir, seeds=(7,11,19,23,31))` returns JSON-serializable results for all methods and k values.

- [x] Add tests that frozen output weights are unchanged and that `k=32` identity compression matches the original predictions.
- [x] Run the focused tests and require a failing identity test before adding evaluation code.
- [x] Implement original, random-neuron, random-orthogonal, PCA, class-separating, and supervised-output subspace methods.
- [x] Use pure between-class directions for `k≤5`; for larger k append PCA residual directions and label the result hybrid.
- [x] Compute loss, accuracy, original prediction agreement, reconstruction MSE, class separation ratio, Pearson distance correlation, and rank distance correlation.
- [x] Re-run focused tests and require PASS.

### Task 3: Run and report the experiment

**Files:**
- Create: `uci_har/representation_compression_results.json`
- Create: `uci_har/representation_compression_analysis.md`

- [x] Run `python -m uci_har.representation_compression_experiment`.
- [x] Verify every method has all six dimensions and five model seeds; verify random methods have twenty draws per seed.
- [x] Identify the smallest k within one percentage point of the original test accuracy for each method.
- [x] Write the comparison table and state whether the hypothesis is supported, weakened, or rejected.
- [x] Run focused tests, `git diff --check`, and a syntax check before completion.

### Task 4: Compare with ordinary quantization

**Files:**
- Create: `uci_har/quantization_vs_compression.py`
- Create: `uci_har/test_quantization_vs_compression.py`
- Create: `uci_har/quantization_vs_compression_results.json`
- Create: `uci_har/quantization_vs_compression_analysis.md`

- [x] Test symmetric per-array INT8/INT4/INT2 dequantization and theoretical payload size.
- [x] Run the quantization comparison on the same five model seeds.
- [x] Compare accuracy, model bytes, hidden activation bytes, and dense/projection MACs.
- [x] Document that post-hoc representation compression does not shrink the original model or dense path.
- [x] Run focused tests, syntax checks, and `git diff --check`.

### Task 5: Train and benchmark a real 8-dimensional bottleneck

**Files:**
- Create: `uci_har/bottleneck_experiment.py`
- Create: `uci_har/test_bottleneck_experiment.py`
- Create: `uci_har/bottleneck_results.json`
- Create: `uci_har/bottleneck_analysis.md`

- [x] Test parameter counting, activation-byte counting, and FLOP formulas.
- [x] Train `561→64→32→6` and `561→64→8→6` from matched seeds with the same strict split, Adam, 80 epochs, batch size 128, and learning rate 0.001.
- [x] Evaluate test accuracy and loss without using test data for training or architecture selection.
- [x] Measure FP32 non-input activation memory per sample, parameter count/storage, median full-test inference time, MACs, and FLOPs.
- [x] Report absolute values and percentage changes, then state whether the real bottleneck preserves the earlier post-hoc compression result.
- [x] Run focused tests, syntax checks, and `git diff --check`.
