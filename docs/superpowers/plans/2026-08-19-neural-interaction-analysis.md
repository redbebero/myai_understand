# Neural Interaction Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure hidden-node interactions in the existing UCI HAR MLP and reconstruct their input-conditional roles as readable rules.

**Architecture:** Add one experiment module beside the existing UCI HAR experiment. It will copy models for ablation, evaluate accuracy/loss/probabilities, rank node pairs by `D(i,j)-D(i)-D(j)`, summarize joint activations and class changes, and fit a small named-role student using existing helpers. Store machine-readable results and a concise interpretation report.

**Tech Stack:** Python, NumPy standard library JSON, unittest.

**Spec:** User-provided active goal in the conversation.

## Global Constraints

- Reuse the existing NumPy-only UCI HAR model and official train/test split.
- Do not add dependencies or alter existing baseline behavior.
- Fix seed, model training settings, evaluation split, and activation threshold in the experiment.
- Treat interaction as evidence of current model behavior, not proof of universal necessity.

---

### Task 1: Define the interaction-analysis API with failing tests

**Files:**
- Create: `uci_har/test_interaction_experiment.py`
- Create: `uci_har/interaction_experiment.py`

**Interfaces:**
- `ablate_hidden_pair(model, layer, first, second) -> dict`
- `evaluate_outputs(model, inputs, targets) -> dict`
- `pair_interactions(model, inputs, targets, layer, candidates) -> list[dict]`
- `joint_activation_summary(model, inputs, targets, layer, first, second, threshold=0.0) -> dict`
- `class_change_summary(model, ablated, inputs, targets) -> dict`

- [ ] **Step 1: Write tests for pair ablation, score calculation, activation counts, and output metrics.**
- [ ] **Step 2: Run `python -m unittest uci_har.test_interaction_experiment -v` and confirm the new imports/functions fail.**
- [ ] **Step 3: Implement model copying, pair zeroing, accuracy/cross-entropy/prediction-probability evaluation, and the interaction formula.**
- [ ] **Step 4: Run the focused tests and confirm they pass.**

### Task 2: Add bounded experiment execution and readable candidates

**Files:**
- Modify: `uci_har/interaction_experiment.py`
- Modify: `uci_har/test_interaction_experiment.py`

**Interfaces:**
- `run_experiment(data_dir, seeds=(7, 11), top_nodes=8, top_pairs=10) -> dict`
- `write_report(result, path) -> None`

- [ ] **Step 1: Add a test that a synthetic small model produces stable JSON-serializable result fields and named rule candidates.**
- [ ] **Step 2: Run the focused test and confirm it fails for the missing runner/report.**
- [ ] **Step 3: Train the existing baseline per seed, select candidate nodes from single-ablation drops, evaluate all candidate pairs, and retain top interaction pairs.**
- [ ] **Step 4: Record joint activation rates by class, prediction flips by class, and candidate rules from existing named feature groups.**
- [ ] **Step 5: Add a minimal student model using existing `role_features` and report teacher accuracy, student accuracy, and teacher agreement.**
- [ ] **Step 6: Run the focused tests and confirm they pass.**

### Task 3: Run the experiment and document evidence

**Files:**
- Create: `uci_har/interaction_results.json`
- Create: `uci_har/interaction_analysis.md`
- Modify: `uci_har/README.md`

- [ ] **Step 1: Run `python -m uci_har.interaction_experiment` with the fixed settings and save JSON output.**
- [ ] **Step 2: Inspect top pairs, joint activation tables, class changes, seed variation, and student comparison.**
- [ ] **Step 3: Write the report with hypothesis, contradiction, revised principle, evidence, failed interpretations, limitations, and reproduction command.**
- [ ] **Step 4: Run all UCI HAR tests and verify existing baseline tests remain green.**

### Task 4: Completion audit

- [ ] **Step 1: Verify the JSON contains individual and joint ablations, interaction scores, activation conditions, class changes, seed comparisons, and student metrics.**
- [ ] **Step 2: Verify the Markdown contains a table of top groups, readable rule candidates, baseline/reconstruction comparison, and limitations.**
- [ ] **Step 3: Run the reproduction command from a clean current worktree state and record the passing test command.**
- [ ] **Step 4: Only after all explicit goal requirements have evidence, mark the goal complete.**
