# Automatic Raw-Structure Discovery Report

## Purpose

This experiment removes the human-defined `energy`, `periodicity`, and `correlation` feature dictionary from the discovery stage.

The pipeline is:

```text
trained MLP
→ walking-vs-upstairs score
→ exact local raw-input influence
→ high-influence channel-time regions
→ primitive expression search on those regions
→ held-out score R²
```

The human role is deferred to the final interpretation of the selected raw expressions.

## Region discovery

For each checkpoint, the algorithm used fitting subjects only to compute mean absolute influence over the `9 × 128` raw coordinates. Within each channel, high-influence timesteps were thresholded and contiguous runs were merged into regions. If no sufficiently long run existed, the strongest fixed-width local window was selected.

No semantic feature family was used to select these regions.

Examples of automatically selected regions, using raw channel indices:

```text
flat seed 7:  channel 6 [63:79], channel 7 [110:126], channel 5 [68:84]
flat seed 11: channel 6 [50:66], channel 7 [98:114],  channel 5 [30:46]
wide seed 19: channel 6 [94:110], channel 2 [76:92], channel 5 [65:81]
```

Channel indices map to the existing raw input order in `experiment.py`; the discovery algorithm itself only operates on the indices and timesteps.

## Expression discovery

For each discovered region, the program generated primitive expressions:

```text
mean(x[channel,start:end])
mean(square(x[channel,start:end]))
mean(abs(diff(x[channel,start:end])))
max(x[channel,start:end])-min(x[channel,start:end])
slope(x[channel,start:end])
```

For overlapping region pairs it also generated:

```text
mean(x1*x2)
corr(x1,x2)
```

Lasso then ranked these expressions by their ability to predict the actual MLP score on held-out validation subjects.

## Score preservation

Mean validation R² across five seeds:

| architecture | mean R² | SD |
|---|---:|---:|
| flat | 0.272 | 0.032 |
| wide | 0.274 | 0.011 |

The automatically discovered expressions explain less score variation than the earlier human-designed 6-family dictionary. That is expected: this first automatic search is deliberately restricted to a small number of influence-derived regions and primitive expressions.

## Representative automatically selected expressions

The exact expressions vary across checkpoints. Examples include:

```text
max(x[6,63:79]) - min(x[6,63:79])
mean(square(x[5,68:84]))
mean(abs(diff(x[5,68:84])))
mean(square(x[0,96:112]))
mean(abs(diff(x[2,45:61])))
```

These were not supplied as semantic candidates. They were generated from raw coordinates after influence-based region discovery. A person may later interpret them as amplitude/range-like or local-change-like patterns, but that interpretation was not used during selection.

## What this establishes

- The model can identify candidate raw channel-time regions before human semantic labeling.
- Primitive expressions over those regions can be ranked against the actual MLP score.
- The automatically discovered expressions are concrete formulas over raw values, not pre-named feature families.

## Current limitation

The first automatic search has lower validation R² than the full human-defined dictionary. It should therefore be treated as a proof of direction, not as the final discovery system.

The main reasons are methodological:

- only the strongest short regions were searched;
- the expression grammar is intentionally small;
- region pairs were limited to overlapping windows;
- the current Lasso explanation remains moderately dense;
- the search selects regions from aggregate influence, which can miss class-conditional or signed patterns.

The next improvement should expand the grammar and use a complexity-penalized search, while retaining the same untouched-test protocol.

## Bounded conclusion

The direction is now reversed correctly:

> The system first finds important raw channel-time regions from the trained model's own influence maps, then searches simple formulas over those regions. Human-readable labels are assigned only after the formulas are discovered.

The current result shows that this automatic route is operational, but its score-preservation performance still needs improvement before it replaces the broader human-defined feature dictionary.
