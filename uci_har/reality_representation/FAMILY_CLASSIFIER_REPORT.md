# Unseen-Test Evaluation of Human Feature Families

## Question

Do the selected human-readable families retain enough information to train a new classifier that predicts unseen test subjects, and how does its accuracy compare with the original raw-input MLP?

## Protocol

The family order was fixed from the validation compression analysis:

```text
temporal change
→ acceleration–rotation coupling
→ movement energy
→ cross-channel coordination
→ temporal periodicity
→ sensor level
```

For each prefix of this order, a new linear softmax classifier was trained from scratch using only those family values. Training used the same fitting subjects as the original MLP; evaluation used the untouched UCI-HAR test subjects. Five classifier seeds were evaluated.

This is a classification test, not an MLP-score R² test.

## Results

| retained families | feature count | unseen-test accuracy |
|---|---:|---:|
| temporal change | 36 | 0.8429 |
| + acceleration–rotation coupling | 72 | 0.8604 |
| + movement energy | 81 | 0.8799 |
| + cross-channel coordination + periodicity + level | 144 | 0.9161 |

Original raw-input MLP references from the same experiment:

| original model | unseen-test accuracy |
|---|---:|
| flat average | 0.8410 |
| wide average | 0.8487 |

The three-family classifier reached `0.8799`, which is approximately:

```text
+0.0389 versus flat MLP average
+0.0312 versus wide MLP average
```

The six-family linear classifier reached `0.9161`.

## Interpretation

The result is stronger than score explanation alone. The selected human-readable families are not merely correlated with the original score; they contain enough information to train a classifier that generalizes to unseen test subjects.

The first three families already perform well:

```text
temporal change
+ acceleration–rotation coupling
+ movement energy
→ 0.8799 test accuracy
```

Adding the remaining three families improves accuracy further:

```text
0.8799 → 0.9161
```

Therefore, for this dataset and split, the three-family representation is a useful compact representation for classification, while the six-family representation retains more discriminative information.

## Important boundary

The three families were selected using validation score-compression results, so this is not a fully nested model-selection estimate. The test set was not used to choose the families or train the classifier, but a stricter future protocol should choose the family order inside an inner validation split and reserve a final test set.

The human-feature classifier and original MLP are also different model classes: a linear softmax classifier versus the original nonlinear MLP. The comparison answers whether the feature representation supports generalization; it does not isolate representation quality from classifier inductive bias.

## Conclusion

For the current UCI-HAR experiment:

> The information families found by reverse-tracing are sufficient to support an independently trained classifier on unseen subjects. Temporal change, acceleration–rotation coupling, and movement energy form a compact useful representation, while adding cross-channel coordination, periodicity, and sensor level recovers additional accuracy.
