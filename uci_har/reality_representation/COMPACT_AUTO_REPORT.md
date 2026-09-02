# Compact AI-only representation search

## Objective

Reduce the automatic representation from 80 expressions to 5–15 expressions while preserving predictive accuracy.

## Method

- Keep only raw channel/time regions discovered by the AI.
- Generate the automatic expression candidates.
- Rank 30 candidates with the largest predictive classifier weights.
- Greedily add one expression at a time using fitting-subject training and validation-subject accuracy.
- Evaluate the selected 5–15-expression representations on untouched test subjects.

## Result

The compact representation did not preserve the 80-expression performance.

| automatic representation | mean test accuracy |
|---|---:|
| previous top-three expressions | 0.7432 |
| expanded 80-expression representation | 0.8039 |
| greedy compact 5–15 expressions | 0.7520 |

The compact runs selected 15 expressions in eight cases, 12 in two cases, and 8 in one case. The test range was **0.7279–0.7879**.

## Interpretation

The result is negative but informative. The extra automatic expressions carry distributed predictive information; removing them causes a substantial loss. The current candidate grammar does not support a 5–15-expression representation with the target performance.

The greedy selector also shows validation overfitting: validation selection did not reliably predict test performance. A single validation split is not sufficient for reliable compact symbolic selection.

## Next technical direction

Use nested subject-level cross-validation for expression selection, then apply stronger symbolic simplification only after identifying a stable expression set. Candidate expressions should also include domain-neutral primitives such as RMS, standard deviation, absolute mean, lagged change, and cross-axis magnitude.

## Reproducibility

```bash
python -m py_compile improved_auto_comparison.py
python improved_auto_comparison.py
```

Output:

```text
results/advanced_trace/improved_auto_results.json
```
