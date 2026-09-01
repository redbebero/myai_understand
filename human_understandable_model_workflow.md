# Human-Understandable Model Reconstruction Workflow

## Purpose

This workflow studies whether a trained model can be understood and rebuilt as a
smaller model whose calculation units have meanings that a person can explain.

The goal is not to claim that every neural node has one perfect human meaning.
The goal is to find evidence-based, interpretable roles that preserve useful
predictions.

## Core question

> After a model learns a difficult pattern, can its weights and node responses
> guide a human to identify important calculations, replace them with explicit
> roles, and retain the model's predictive ability?

## Overall flow

```text
choose a difficult task
        ↓
train an ordinary baseline model
        ↓
save weights, data, outputs, and metrics
        ↓
inspect weights and intermediate activations
        ↓
remove or intervene on nodes and connections
        ↓
measure importance and interaction
        ↓
propose human-readable calculation roles
        ↓
replace selected learned calculations with fixed roles
        ↓
learn only the remaining combination parameters
        ↓
compare accuracy, size, speed, and explanations
        ↓
repeat with new data and new training conditions
```

## 1. Choose the task

Use a task where writing one obvious rule beforehand is difficult, but where
inputs and outputs can still be measured clearly.

Good examples include:

- noisy multivariate time-series classification;
- activity or event recognition from several sensors;
- simple image or shape classification;
- anomaly detection in multiple signals;
- sound or frequency-pattern classification.

Avoid a task whose answer is already known as a short formula if the purpose is
to study what the model learned. A known formula can be used as a control
experiment, but it must be separated from weight-based reconstruction.

## 2. Build and train the baseline

Start with a small ordinary model. The person chooses only the general input,
output, and capacity constraints. The model learns its internal weights.

Save:

- training, validation, test, and fresh-data splits;
- architecture;
- all weights and biases;
- random seed and training settings;
- accuracy, loss, and prediction probabilities.

The baseline is the reference. Every redesign must be compared with it.

## 3. Inspect weights without assigning meanings too early

For each node, inspect:

- which input channels have large weights;
- whether weights focus on a time region or are distributed;
- positive and negative weight groups;
- bias and activation range;
- connections entering and leaving the node;
- repeated patterns across nodes.

Weights are evidence about sensitivity, not labels. A large weight does not by
itself prove that a node means “motion,” “edge,” or “risk.” The complete pathway
and observed behavior must also be checked.

## 4. Measure node importance by intervention

Remove one component at a time and compare performance:

```text
importance = baseline performance - performance after removal
```

Test:

- individual connections;
- individual nodes;
- groups of similar nodes;
- input features;
- intermediate calculations.

Also compare the model's raw output, not only its final class. A component can
leave accuracy unchanged while still changing confidence or decision boundaries.

Importance has two meanings that must be separated:

- **current-use importance:** the original model currently relies on it;
- **structural necessity:** no equivalent smaller structure can replace it.

Removal alone proves the first, not the second.

## 5. Observe node responses

Run many inputs through the trained model and record intermediate activations.
For each node, create a response profile:

- mean and variance of activation;
- activation by class or condition;
- activation when one input channel changes;
- activation when time order changes;
- activation under noise or scaling;
- activation before and after controlled interventions.

The question is:

> Which measurable property of the input changes when this node changes?

## 6. Create candidate human-readable roles

A role is an explicit calculation that summarizes one interpretable property of
the input. Candidate roles depend on the domain.

Examples:

- signal magnitude;
- energy or RMS;
- average level;
- change between adjacent time points;
- trend;
- variance;
- channel difference;
- frequency-band strength;
- local contrast;
- edge direction;
- count, threshold, or duration.

The human proposes candidates as a vocabulary for interpretation. The model's
weights and activations provide evidence for or against each candidate.

## 7. Match roles to learned behavior

Compare each candidate role with each node's activation using correlation,
regression, mutual information, or controlled intervention.

For a candidate role and node, record:

- strength of association;
- positive or negative direction;
- performance when the role is removed;
- performance when the role is replaced by noise;
- whether the association repeats across random seeds.

Use careful language:

```text
weak: the node has a large weight on channel 2
stronger: the node activation correlates with channel-2 energy
strongest: replacing the node with channel-2 energy preserves behavior
```

Do not convert correlation directly into certainty.

## 8. Redesign the model

Replace selected learned nodes with fixed, named calculations:

```text
raw input
→ fixed interpretable roles
→ learned combination layer
→ prediction
```

The fixed roles should be explicit enough that a person can calculate them by
hand or implement them without a neural-network library.

Only the remaining combination parameters are learned. This creates a fair
middle ground between two extremes:

```text
fully hand-written algorithm: structure and weights fixed by a person
ordinary neural network: structure and weights learned
interpretable hybrid: roles fixed by a person, combination learned
```

Record exactly which quantities are fixed and which are trainable.

## 9. Verify the redesign

Compare the baseline and redesign on:

- training data;
- held-out test data;
- fresh data from the same process;
- changed noise levels;
- changed input scale;
- different random seeds;
- shifted or new conditions.

Measure:

- predictive accuracy or error;
- confidence/output difference;
- parameter count;
- node and connection count;
- file size;
- runtime;
- memory use;
- explanation length and clarity.

A redesign is useful if it is smaller and understandable while retaining
performance on data it did not see. Matching the training set alone is weak
evidence.

## 10. Test interactions and redundancy

Nodes may work as a group. After finding important individual nodes, test:

- two-node removal;
- group removal;
- replacement of a group by one role;
- retraining after removal;
- whether another node compensates after retraining.

This distinguishes a node that is important only in the current representation
from a calculation that is genuinely necessary for the task.

## 11. Keep three claims separate

Every report should distinguish:

### Exact reconstruction

The new model produces the same function for every relevant input.

### Behavioral imitation

The new model produces similar predictions or classes on tested data.

### Human redesign

The new model uses roles chosen from analysis and is easier to explain, even if
it is not mathematically identical to the original.

Most practical experiments establish the second or third claim, not the first.

## 12. Record limitations

Always state:

- whether the data-generation rule was known;
- whether role candidates were chosen by a person;
- whether the model was retrained;
- whether only one seed was tested;
- whether correlation was mistaken for causation;
- whether the redesign reproduces outputs or only labels;
- whether the task is synthetic or real-world.

## Final research interpretation

The strongest conclusion is not:

> I discovered the one true meaning of every neural node.

It is:

> I used learned weights, node interventions, and activation responses to infer
> candidate computational roles. I then replaced part of the learned structure
> with explicit human-readable calculations and tested how much predictive
> behavior remained. The results reveal both what can be simplified and which
> interactions resist human compression.

This keeps the work focused on curiosity, interpretation, redesign, and
verification rather than on merely increasing or decreasing model size.
