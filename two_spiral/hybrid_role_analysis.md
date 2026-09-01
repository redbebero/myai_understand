# Fixed Roles, Partly Learned Weights

## The distinction

If a person chooses the node equations **and** every numerical coefficient, the
result is a hand-written algorithm. It is not a learned model:

```text
person chooses structure + person chooses weights = algorithm
```

The hybrid model separates those decisions:

```text
person chooses node meanings and equations
machine learns only how strongly to combine their outputs
```

## Fixed node layer

The role layer is fixed as:

```text
r = hypot(x, y)
theta = atan2(y, x)
phase = theta - 4πr
arm_0 = cos(phase)
arm_1 = cos(phase - π)
```

It also exposes distance and direction indicators. These equations never change
during training. Only the final linear-logistic vote changes:

\[
z=\sum_{i=0}^{6}w_i\phi_i(x,y)
\]

\[
p=\frac{1}{1+e^{-z}}
\]

There are 7 trainable combination weights and no trainable role-node weights.

## Results

| Model | Role equations | Combination weights | Train | Test |
|---|---|---|---:|---:|
| Fully hand-designed | manual | manual | 95.8% | 97.5% |
| Hybrid | manual | learned, 7 values | 96.7% | 97.5% |

The hybrid learned these final weights:

```text
[-4.0166, 3.0780, -0.3723, 1.4001, 0.6043, -2.5707, 2.5707]
```

The two arm weights became equal and opposite, which matches the intended role:
the final decision mainly compares the two opposing spiral responses.

## Meaning

This is not the same as ordinary neural-network training. A normal network must
discover both:

```text
what each node detects
how nodes should be combined
```

The hybrid model receives the first answer from the human and learns only the
second. This makes the model more interpretable, but its success depends on the
human choosing useful roles. If the spiral-phase node is removed, the remaining
distance and direction information cannot separate the classes.

## Reproduction

```bash
cd two_spiral
python hybrid_role_model.py
python -m unittest test_hybrid_role_model.py test_human_role_model.py -v
```
