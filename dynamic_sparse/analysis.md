# Dynamic Sparse Training Result

## Question

Can an AI learn while connections are removed and regrown, instead of keeping every possible connection throughout training?

## Setup

- Task: two-spiral classification
- Architecture: `2 → 12 → 12 → 1`
- Possible weights: 180
- Active weights: 90 (50%)
- Rewiring: every 100 epochs
- At each rewiring: remove one active connection with the smallest absolute weight in each layer; grow one inactive connection with the largest accumulated gradient in each layer

## Interpretation

The model learns both values and topology. A zero mask means the connection is not part of the computation. A rewiring event changes which calculations are available for later learning.

Compare `result.json` with `../two_spiral/spiral_analysis.json`. The important comparison is not only accuracy. It is whether a model with fewer active connections can reach similar accuracy, and whether the same connections survive across random seeds.

This run reached `58.3%` training accuracy and `52.5%` test accuracy after 900 epochs. It learned only slightly above chance, so this particular rewiring rule did not match the dense baseline. That is a useful result: removing and regrowing connections is not automatically better; the timing and growth rule matter.

## Limitation

This is a small experiment with a simple rewiring rule. It cannot establish that dynamic sparse training is generally better. It tests whether changing topology during learning is worth investigating for the question: which computational connections are actually needed?
