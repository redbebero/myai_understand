# Human-Designed versus AI-Discovered Representation

## Fair comparison protocol

All methods were trained on the same fitting subjects and evaluated on the untouched UCI-HAR test subjects.

### Human-designed representation

The human method supplies predefined sensor summaries and groups them into the selected top-three or all-six families. A new linear softmax classifier is trained on those values.

### AI-discovered representation

The AI method does not receive semantic feature families. It uses raw-only learned bottleneck checkpoints, extracts the top three automatically selected raw expressions for each of four latent dimensions, and trains a new linear softmax classifier using those expressions.

The AI expression set was selected from fitting data and latent validation mapping; the test set was not used to select formulas.

## Test accuracy

| representation | mean unseen-test accuracy |
|---|---:|
| human top 3 families | 0.8799 |
| human all 6 families | 0.9161 |
| AI-discovered raw expressions | 0.7432 |

The AI expression representation used an average of `11.7` expressions per checkpoint, because duplicate formulas were removed across the four latent mappings.

## What the comparison means

The human-designed feature representation currently performs better as a directly trainable classifier input. The first AI-discovered symbolic representation loses information when converted from the learned latent/raw mapping into a small expression set.

This does **not** show that AI discovery is inferior in principle. The compared representations are not equally mature:

- the human representation was designed specifically as a complete classification feature set;
- the AI representation was optimized to approximate latent coordinates, not directly to maximize classifier accuracy;
- the AI representation uses a restricted expression grammar and only the top three expressions per latent;
- latent coordinates are not fully aligned across models;
- the raw gate did not produce a validated sparse coordinate subset.

The correct current conclusion is:

> Human feature engineering still gives the stronger usable classifier representation in this experiment. AI-only bottleneck learning successfully discovers a compact predictive latent space, but the current automatic raw-expression translation does not yet preserve enough information to match the human-designed classifier.

## What would count as success

For AI discovery to replace or match manual analysis, the following must be demonstrated on the same split:

```text
AI-discovered representation accuracy >= human representation accuracy
with fewer numbers, lower or comparable expression complexity,
and stable structures across seeds and architectures.
```

Until then, AI discovery is a discovery aid rather than a validated replacement for human feature engineering.
