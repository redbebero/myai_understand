# Improved AI-only representation

## Change

The first automatic comparison selected the top three expressions per latent using latent reconstruction. This experiment keeps the raw channel/time regions discovered by AI, expands the candidate pool to 80 expressions, and selects the number of expressions using validation-set classification accuracy.

The test set is untouched during selection. A new linear softmax classifier is fitted on the fitting subjects for each selected representation.

## Results

| representation | mean test accuracy |
|---|---:|
| previous AI-only expressions | 0.7432 |
| improved AI-only expressions | 0.8039 |
| human top three families | 0.8799 |
| human all six families | 0.9161 |

The improved automatic representation gains **6.07 percentage points** over the previous automatic representation. Across ten runs, test accuracy ranges from **0.7747 to 0.8290**.

The selected representation contains 80 expressions in every run. This improves predictive accuracy but is not yet a compact human measurement.

## Interpretation

The main bottleneck was expression selection, not only latent dimensionality. Selecting expressions for the downstream classification objective recovers information that was lost when retaining only the top three expressions per latent.

The result is still below the human-designed representation:

- 7.61 percentage points below the human top-three-family representation.
- 11.23 percentage points below the complete human representation.

This is an honest intermediate result: AI-discovered raw regions contain useful predictive information, but the current expression grammar and selection method do not yet produce a smaller or stronger representation than human feature engineering.

## Reproducibility

```bash
python -m py_compile improved_auto_comparison.py
python improved_auto_comparison.py
```

Outputs:

```text
results/advanced_trace/improved_auto_results.json
```
