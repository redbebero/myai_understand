# Raw-sensor experiment record

## Question

Can a model learn temporal sensor calculations from raw signals, and can those
calculations be replaced by named human-readable roles while retaining most of
the model's activity-recognition ability?

## Baseline structure

```text
9 raw channels × 128 time steps
→ 12 learned filters, kernel size 9
→ ReLU
→ temporal mean
→ 6-class softmax
```

### Added 12-node human-readable model

The 57 fixed role calculations were followed by 12 learned ReLU nodes—the
same node count as the original CNN filters—and then the 6-class output layer.

| model | trainable parameters | model file | test accuracy |
|---|---:|---:|---:|
| learned raw CNN: 9×128 → 12 filters → 6 | 1,062 | 9,502 B | 92.13% |
| fixed roles: 57 → 6 | 348 | 4,686 B | 88.36% |
| fixed roles → 12 learned nodes → 6 | 774 | 8,620 B | 85.75% |

The extra 12 nodes did not recover the lost accuracy in this first training
run. Matching the node count is not enough: the fixed roles discard local
timing patterns, while the original filters see short temporal windows.

### Full proposed calculation experiment

The next redesign computed all proposed local and relational quantities: 8
time windows, 9 channels, 12 named per-channel calculations, 6 cross-sensor
calculations per window, and 3 global calculations. This produced 915 named
calculation values. They were grouped into 12 human-readable role nodes and a
12-to-6 classifier was trained.

| model | human-readable calculation stage | role nodes | trainable parameters | test accuracy |
|---|---|---:|---:|---:|
| learned raw CNN | none; raw temporal filters | 12 learned filters | 1,062 | 92.13% |
| fixed roles | 57 global sensor calculations | 0 hidden roles | 348 | 88.36% |
| roles + matching hidden layer | 57 calculations | 12 learned nodes | 774 | 85.75% |
| expanded grouped roles | 915 local/relational calculations | 12 named roles | 78 | 72.01% |
| temporal relational roles | 8 ordered windows × named roles and sensor relations | 147 named values | 888 | 81.40% |

The expanded model is intentionally a meaningful negative result. Grouping the
calculations by averaging across windows destroys the order of events. It
shows that adding more statistics is not enough; the redesign must preserve
which event happened first, how long it lasted, and which sensors changed
together.

The temporal relational model preserves the order of the 8 windows and adds
the cross-sensor values, recovering accuracy from 72.01% to 81.40%. It still
averages the nine channels for each named role, so sensor-specific combinations
remain lost. This is evidence that temporal order matters, but it is not yet
an exact reconstruction of the CNN.

### Structured threshold-role model

The next implementation directly used the mathematical improvement: 8 ordered
windows, 6 named roles per sensor, 12 fixed sensor pairs with 3 relations each,
and 6 event roles. It then added an interpretable threshold feature for each
event role: a value contributes only when it is above the training 75th
percentile. The classifier learns only the final role-to-activity weights.

| model | raw role values | threshold features | trainable parameters | file | test accuracy |
|---|---:|---:|---:|---:|---:|
| raw CNN | 0 | 0 | 1,062 | 9,502 B | 92.13% |
| existing fixed roles | 57 | 0 | 348 | 4,686 B | 88.36% |
| structured threshold roles | 768 | 48 | 4,902 | 53,130 B | 85.92% |

This model is more faithful to the CNN's local and nonlinear calculation, but
it is not yet more accurate or smaller. The reason is that its fixed role set
still assumes which 12 sensor pairs matter and uses one global classifier over
all role values. The CNN learns both the pair selection and the local pattern
shape. The next improvement would be sparse, named role-to-class connections
or a small learned temporal combiner, while keeping every connection's role
explicit.

### Required comparison: readable redesign versus quantization

To separate interpretability from compression, the same raw CNN weights were
also quantized without changing the topology. The comparison used the held-out
UCI HAR test set. Runtime is a NumPy CPU measurement for this implementation,
not a hardware-optimized int8 benchmark.

| model | test accuracy | parameters | file | test inference |
|---|---:|---:|---:|---:|
| raw CNN | 92.13% | 1,062 | 9,502 B | 562 ms |
| 8-bit weight quantization | 91.99% | 1,062 | theoretical 1,062 B weights | 560 ms |
| 4-bit weight quantization | 91.82% | 1,062 | theoretical 531 B weights | 560 ms |
| human 57-role model | 88.36% | 348 | 4,686 B | 851 ms |
| structured readable model | 85.92% | 4,902 | 53,130 B | 9,811 ms |

The result answers an important part of the research question: the readable
models are not currently a better compression or speed method than
quantization. Eight-bit quantization preserves nearly all behavior because it
keeps the original learned computation. The readable models change the
computation itself, so they expose meaning but lose some accuracy and, when
they compute many statistics, become slower.

The model has 1,062 trainable parameters and was implemented with NumPy only.
The UCI subject-based train/test split is preserved, so the test set contains
people not used for training.

## Human-readable roles

For each of the 9 channels, the redesign calculates:

- mean;
- standard deviation;
- RMS energy;
- mean absolute step change;
- start-to-end trend;
- strongest non-constant frequency component.

It also calculates global RMS, global mean absolute change, and cross-channel
variation. These 57 role values are fixed. Only the final 57-to-6 softmax
weights are learned, giving 348 trainable parameters.

## Result

```text
                         train       test       noise 0.05   scale 1.1
learned raw CNN           95.73%      92.13%      91.89%      91.96%
fixed-role student        90.38%      88.36%      87.24%      87.92%
```

The role model retains 95.9% of the baseline test accuracy and uses 32.8% of
the parameters. It is behavioral imitation and human redesign, not exact
reconstruction of the CNN function.

The baseline was also retrained with seeds 11 and 19. Their train/test scores
are stored in `results.json` so the conclusion is not based on one lucky
initialization.

## What the internal analysis found

The script records, in `analysis.json`:

- the accuracy drop after removing each filter;
- mean filter activation and class-wise responses;
- correlation between each learned filter and each named role;
- accuracy after removing the two most important filters together;
- accuracy after zeroing each sensor channel.

A strong role match is evidence that a filter responds to that quantity. It is
not proof that the filter has one unique meaning. The filter-removal result is
the intervention evidence; the correlation is only the interpretation aid.

## Relation to the research question

This is stronger than the previous 561-feature experiment because the baseline
does not receive precomputed statistical features. It learns from raw temporal
signals first. The gap between 92.13% and 88.36% is useful evidence: simple
single-channel roles explain much of the decision, but some learned temporal
interactions are not captured by this first redesign.

That gap is not a failure. It identifies the next research question:

> Which combinations of human-readable roles are required to replace the
> temporal interactions represented by the remaining filters?

## Claims kept separate

- Exact reconstruction: not demonstrated.
- Behavioral imitation: demonstrated on the held-out test split with lower
  accuracy than the teacher.
- Human redesign: demonstrated by replacing learned filters with 57 fixed,
  named calculations and learning only their final combination.

## Limitations

- The role-model redesign itself uses one seed; the baseline stability check
  uses three seeds.
- Role candidates were chosen by the researcher before matching.
- The convolution is a compact baseline, not a state-of-the-art HAR model.
- Correlation cannot establish causality by itself.
