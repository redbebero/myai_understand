# Reality Representation Experiment

## Protocol

- **input:** 9 raw channels x 128 timesteps
- **validation_subjects:** [1, 3, 5, 6, 7, 8]
- **relation_breaking:** within-sample channel permutation
- **variants:** ['flat', 'wide']
- **seeds:** [7, 11, 19, 23, 31]

## Results

| variant | mean baseline | mean relation-broken | mean drop | mean intervention | mean control |
|---|---:|---:|---:|---:|---:|
| flat | 0.8410 | 0.5035 | 0.3375 | 0.0458 | 0.0790 |
| wide | 0.8487 | 0.5129 | 0.3358 | 0.0364 | 0.0732 |

## Recurring candidate features

- body_acc_x:energy
- body_acc_x:variation
- body_acc_x×body_acc_y:coupling
- body_acc_x×total_acc_x:coupling
- body_acc_x×total_acc_y:coupling
- body_acc_x×total_acc_z:coupling
- body_acc_y:variation
- body_acc_y×total_acc_y:coupling
- body_gyro_x:energy
- body_gyro_x:variation
- body_gyro_x×body_gyro_y:coupling
- body_gyro_y:variation
- body_gyro_y×body_gyro_z:coupling
- body_gyro_z:energy
- body_gyro_z:variation
- total_acc_x:energy
- total_acc_x:level
- total_acc_x:slope
- total_acc_x:variation
- total_acc_x×total_acc_z:coupling
- total_acc_y:energy
- total_acc_y:level
- total_acc_y:variation
- total_acc_y×total_acc_z:coupling
- total_acc_z:energy
- total_acc_z:level
- total_acc_z:variation

## Human-readable interpretation cards

### flat / seed 7
- Direction 0: **temporal change pattern**; strongest pair effect: `sitting vs standing`; required information: total_acc_y:variation, body_gyro_y:variation, body_acc_y:variation
- Direction 1: **movement intensity**; strongest pair effect: `sitting vs laying`; required information: total_acc_x:level, total_acc_x:energy, total_acc_y:level
- Direction 2: **temporal change pattern**; strongest pair effect: `walking vs walking_upstairs`; required information: total_acc_x:variation, body_acc_x:variation, body_acc_x:energy
- Direction 3: **temporal change pattern**; strongest pair effect: `walking_upstairs vs walking_downstairs`; required information: body_gyro_y:variation, total_acc_z:variation, body_acc_z:variation
- Direction 4: **cross-channel coordination**; strongest pair effect: `walking vs walking_upstairs`; required information: total_acc_y:energy, total_acc_x×total_acc_y:coupling, total_acc_y:level
- Direction 5: **temporal change pattern**; strongest pair effect: `walking vs laying`; required information: body_gyro_z:variation, total_acc_z:variation, total_acc_y:variation
### flat / seed 11
- Direction 0: **linear-motion and rotation coupling**; strongest pair effect: `sitting vs standing`; required information: total_acc_z:level, total_acc_x×total_acc_z:coupling, body_gyro_y×total_acc_x:coupling
- Direction 1: **movement intensity**; strongest pair effect: `walking vs sitting`; required information: total_acc_x:energy, total_acc_x:level, total_acc_y:energy
- Direction 2: **temporal change pattern**; strongest pair effect: `walking vs walking_upstairs`; required information: body_gyro_z:variation, body_acc_y:variation, total_acc_y:variation
- Direction 3: **cross-channel coordination**; strongest pair effect: `walking_upstairs vs walking_downstairs`; required information: total_acc_x:energy, total_acc_x:level, total_acc_x×total_acc_y:coupling
- Direction 4: **movement intensity**; strongest pair effect: `walking vs walking_downstairs`; required information: total_acc_x:level, total_acc_y:level, total_acc_x:energy
- Direction 5: **cross-channel coordination**; strongest pair effect: `walking_upstairs vs laying`; required information: total_acc_z:energy, total_acc_y×total_acc_z:coupling, total_acc_y:energy

## Interpretation boundary

Relation destruction supports dependence on temporal/channel structure. Human-readable cards describe candidate abstractions; causal claims require selective intervention beyond the current permutation control.
