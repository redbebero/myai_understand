# Symbolic Explanation Aggregation

Automatically discovered expressions were parsed, mapped to raw channel names and time windows, then clustered only when operation, channels, and windows matched approximately.

| readable concept | raw structure | model recurrence | mean coefficient |
|---|---|---:|---:|
| body_gyro_z local signal magnitude | `local_squared_magnitude` on body_gyro_z[68:84] | 5/10 | -0.148 |
| total_acc_x local movement range | `local_range` on total_acc_x[94:110] | 3/10 | -0.149 |
| body_acc_x local signal magnitude | `local_squared_magnitude` on body_acc_x[38:54] | 3/10 | -0.138 |
| body_gyro_z local movement range | `local_range` on body_gyro_z[74:90] | 3/10 | 0.131 |
| body_gyro_z local signal magnitude | `local_squared_magnitude` on body_gyro_z[85:101] | 3/10 | -0.125 |
| body_gyro_z local movement range | `local_range` on body_gyro_z[85:101] | 2/10 | 0.161 |
| total_acc_x local movement range | `local_range` on total_acc_x[63:79] | 2/10 | -0.158 |
| body_gyro_z local movement range | `local_range` on body_gyro_z[30:46] | 2/10 | 0.155 |
| body_acc_z local temporal change | `local_temporal_change` on body_acc_z[45:61] | 2/10 | 0.154 |
| body_acc_z local temporal change | `local_temporal_change` on body_acc_z[76:92] | 2/10 | 0.145 |
| body_acc_x local signal magnitude | `local_squared_magnitude` on body_acc_x[96:112] | 2/10 | -0.141 |
| total_acc_y local mean level | `local_mean_level` on total_acc_y[67:83] | 2/10 | 0.127 |

## Interpretation boundary

The readable meaning is generated from the algebraic operator and raw channel identity. It is a label for the discovered computation, not a claim about biological causality. Region clustering uses only raw coordinate overlap and does not impose the earlier human-defined feature families.
