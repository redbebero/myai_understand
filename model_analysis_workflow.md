# Model Analysis Workflow

## 1. Define the question

Choose a problem that is difficult to solve by writing rules directly.

## 2. Build a small experiment

Define the inputs, correct answers, and evaluation method. Build the smallest useful model.

## 3. Save the baseline

Record the model before and after training.

Save:

- Model structure
- Weights and biases
- Training data
- Baseline accuracy or error

## 4. Inspect the computation

Trace how inputs change as they pass through each part of the model.

Inspect:

- Intermediate outputs
- Final outputs
- Predictions for each input

## 5. Remove one part at a time

Remove one connection, neuron, layer, or calculation at a time.

Record what was removed and how the structure changed.

## 6. Measure the performance change

Compare accuracy or error before and after the removal.

```text
importance = performance before removal - performance after removal
```

## 7. Interpret the result

Separate elements that cause a large performance drop from elements that have little effect.

Do not judge importance from the absolute value of a weight alone. Consider the complete structure and interactions with other calculations.

## 8. Retrain the simplified model

Train the model again after removing an element.

Check whether performance is still maintained. If it is, the removed element may not be essential for that problem.

## 9. Repeat the verification

Check whether the same result appears with different initial weights, data orders, and subsets of the data.

Do not treat one experiment as a general law.

## 10. Record and compare experiments

For each experiment, record:

- Research question
- Data used
- Model structure
- Element changed
- Performance before and after the change
- Observed computation
- Interpretation and limitations
