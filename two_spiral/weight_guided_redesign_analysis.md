# Weight-Guided Human Redesign

## What this experiment does

This is the experiment that most directly matches the original question. It
does not use the spiral-generation equation to invent a solution.

1. Start with the trained `2 -> 12 -> 12 -> 1` network.
2. Remove each first-layer neuron one at a time.
3. Measure the accuracy drop.
4. Treat the largest drops as evidence that those neurons matter to this model.
5. Copy only the selected neurons' learned equations into a smaller, readable
   first layer.
6. Freeze those equations.
7. Retrain only the downstream connections.

For a selected neuron, the readable equation is:

\[
n_i(x,y)=\tanh(a_ix+b_iy+c_i)
\]

The redesign therefore preserves the human-readable input tests discovered in
the weights, while allowing the later connections to adapt to the reduced set.

## Weight-only importance result

The six highest-impact first-layer neurons were:

```text
hidden1[3], hidden1[4], hidden1[1], hidden1[2], hidden1[6], hidden1[5]
```

Removing `hidden1[3]` caused the largest training-accuracy drop: 39.2
percentage points. This is evidence from the trained model, not a role assigned
from the spiral formula.

## Redesign result

| Model | First-layer nodes | Trainable parameters | Train | Test |
|---|---:|---:|---:|---:|
| Original network | 12 | 205 total | 94.2% | 87.5% |
| Weight-guided redesign | 6 | 97 downstream/visible parameters | 94.2% | 84.2% |

The smaller redesigned model retained most of the training behavior but lost
3.3 percentage points on unseen test data. The size sweep is in
`weight_guided_redesign_results.json`; it checks whether retaining more selected
nodes recovers the held-out performance.

## Interpretation

This experiment is not a complete success, and that is useful. It shows that a
neuron can be important when removed from the original network but still be
insufficient as part of a smaller redesigned network. Importance depends on the
other connections around the neuron. The model uses distributed computation,
not six independent rules that can simply be lifted out.

The defensible conclusion is:

> Learned weights can identify candidate computational units and provide exact,
> readable node equations. However, redesigning those units requires retraining
> their connections, and removing units can change the representation enough to
> reduce generalization.

This is different from the earlier hand-designed polar model. The polar model
used knowledge of how the data was generated. This experiment uses the trained
weights and ablation results as its starting evidence.
