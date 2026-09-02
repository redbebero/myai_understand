# Complete Human Interpretation of AI-Discovered Latent Values

This report translates every latent mapping's top three automatically selected expressions. The expressions were generated from raw channel-time regions; the human-readable wording is applied only after selection.

## Overall result

- bottleneck dimension: **k=4**
- latent subspace similarity proxy: mean **0.574**, minimum **0.391**
- latent-to-formula validation R²: mean **0.779**, range **0.387–0.968**

## Recurring raw structures

| operation + channels | recurrence among 40 latent mappings | automatic interpretation |
|---|---:|---|
| `local_mean_level` on total_acc_x | 22/40 | local mean level |
| `local_mean_level` on total_acc_y | 22/40 | local mean level |
| `local_squared_magnitude` on body_acc_x | 13/40 | local signal magnitude |
| `local_squared_magnitude` on total_acc_x | 12/40 | local signal magnitude |
| `local_range` on body_acc_x | 10/40 | local movement range |
| `local_squared_magnitude` on total_acc_y | 8/40 | local signal magnitude |
| `local_mean_level` on total_acc_z | 7/40 | local mean level |
| `local_range` on total_acc_x | 6/40 | local movement range |
| `local_squared_magnitude` on total_acc_z | 5/40 | local signal magnitude |
| `local_temporal_change` on body_gyro_y | 4/40 | local temporal change |
| `local_range` on body_gyro_z | 2/40 | local movement range |
| `joint_signal_strength` on total_acc_x, total_acc_y | 2/40 | joint signal strength |
| `local_range` on total_acc_z | 1/40 | local movement range |
| `local_temporal_change` on body_gyro_z | 1/40 | local temporal change |
| `local_range` on total_acc_y | 1/40 | local movement range |
| `joint_signal_strength` on total_acc_y, total_acc_x | 1/40 | joint signal strength |
| `joint_signal_strength` on total_acc_x, total_acc_z | 1/40 | joint signal strength |
| `local_temporal_change` on total_acc_z | 1/40 | local temporal change |
| `local_range` on body_acc_z | 1/40 | local movement range |

## Every latent mapping

### flat seed 7 — latent dimension z1

Validation R² for this latent: **0.890**

1. Formula: `mean(x[6,81:97])`
   - Raw location: `total_acc_x[81:97]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_x local mean level**
   - Lasso coefficient: `8.1860`

2. Formula: `mean(square(x[6,81:97]))`
   - Raw location: `total_acc_x[81:97]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_x local signal magnitude**
   - Lasso coefficient: `-3.9381`

3. Formula: `mean(x[7,6:22])`
   - Raw location: `total_acc_y[6:22]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_y local mean level**
   - Lasso coefficient: `2.4859`

### flat seed 7 — latent dimension z2

Validation R² for this latent: **0.387**

1. Formula: `mean(square(x[0,71:87]))`
   - Raw location: `body_acc_x[71:87]` (0.32 s window)
   - Automatic structural interpretation: **body_acc_x local signal magnitude**
   - Lasso coefficient: `2.2012`

2. Formula: `max(x[8,5:21])-min(x[8,5:21])`
   - Raw location: `total_acc_z[5:21]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_z local movement range**
   - Lasso coefficient: `-1.1856`

3. Formula: `max(x[0,71:87])-min(x[0,71:87])`
   - Raw location: `body_acc_x[71:87]` (0.32 s window)
   - Automatic structural interpretation: **body_acc_x local movement range**
   - Lasso coefficient: `-0.9984`

### flat seed 7 — latent dimension z3

Validation R² for this latent: **0.753**

1. Formula: `mean(square(x[6,16:32]))`
   - Raw location: `total_acc_x[16:32]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_x local signal magnitude**
   - Lasso coefficient: `2.2907`

2. Formula: `mean(x[7,47:63])`
   - Raw location: `total_acc_y[47:63]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_y local mean level**
   - Lasso coefficient: `-1.8306`

3. Formula: `mean(x[6,16:32])`
   - Raw location: `total_acc_x[16:32]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_x local mean level**
   - Lasso coefficient: `-1.8070`

### flat seed 7 — latent dimension z4

Validation R² for this latent: **0.771**

1. Formula: `mean(x[7,83:99])`
   - Raw location: `total_acc_y[83:99]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_y local mean level**
   - Lasso coefficient: `1.9522`

2. Formula: `mean(square(x[6,94:110]))`
   - Raw location: `total_acc_x[94:110]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_x local signal magnitude**
   - Lasso coefficient: `-1.9150`

3. Formula: `mean(square(x[7,83:99]))`
   - Raw location: `total_acc_y[83:99]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_y local signal magnitude**
   - Lasso coefficient: `-1.6837`

### flat seed 11 — latent dimension z1

Validation R² for this latent: **0.649**

1. Formula: `mean(x[7,106:122])`
   - Raw location: `total_acc_y[106:122]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_y local mean level**
   - Lasso coefficient: `1.0779`

2. Formula: `mean(square(x[8,103:119]))`
   - Raw location: `total_acc_z[103:119]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_z local signal magnitude**
   - Lasso coefficient: `0.9126`

3. Formula: `mean(abs(diff(x[5,64:80])))`
   - Raw location: `body_gyro_z[64:80]` (0.32 s window)
   - Automatic structural interpretation: **body_gyro_z local temporal change**
   - Lasso coefficient: `-0.7233`

### flat seed 11 — latent dimension z2

Validation R² for this latent: **0.968**

1. Formula: `mean(x[6,63:79])`
   - Raw location: `total_acc_x[63:79]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_x local mean level**
   - Lasso coefficient: `4.3528`

2. Formula: `mean(x[8,36:52])`
   - Raw location: `total_acc_z[36:52]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_z local mean level**
   - Lasso coefficient: `-1.8465`

3. Formula: `mean(square(x[7,89:105]))`
   - Raw location: `total_acc_y[89:105]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_y local signal magnitude**
   - Lasso coefficient: `-1.7809`

### flat seed 11 — latent dimension z3

Validation R² for this latent: **0.882**

1. Formula: `mean(square(x[7,60:76]))`
   - Raw location: `total_acc_y[60:76]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_y local signal magnitude**
   - Lasso coefficient: `-1.9842`

2. Formula: `mean(x[8,19:35])`
   - Raw location: `total_acc_z[19:35]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_z local mean level**
   - Lasso coefficient: `-1.6813`

3. Formula: `mean(square(x[8,19:35]))`
   - Raw location: `total_acc_z[19:35]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_z local signal magnitude**
   - Lasso coefficient: `-1.5320`

### flat seed 11 — latent dimension z4

Validation R² for this latent: **0.637**

1. Formula: `mean(square(x[0,73:89]))`
   - Raw location: `body_acc_x[73:89]` (0.32 s window)
   - Automatic structural interpretation: **body_acc_x local signal magnitude**
   - Lasso coefficient: `4.8183`

2. Formula: `mean(x[7,85:101])`
   - Raw location: `total_acc_y[85:101]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_y local mean level**
   - Lasso coefficient: `2.9589`

3. Formula: `mean(square(x[7,85:101]))`
   - Raw location: `total_acc_y[85:101]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_y local signal magnitude**
   - Lasso coefficient: `-2.3151`

### flat seed 19 — latent dimension z1

Validation R² for this latent: **0.761**

1. Formula: `mean(x[7,62:78])`
   - Raw location: `total_acc_y[62:78]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_y local mean level**
   - Lasso coefficient: `2.7757`

2. Formula: `max(x[6,12:28])-min(x[6,12:28])`
   - Raw location: `total_acc_x[12:28]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_x local movement range**
   - Lasso coefficient: `-2.4679`

3. Formula: `mean(x[6,12:28])`
   - Raw location: `total_acc_x[12:28]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_x local mean level**
   - Lasso coefficient: `1.5349`

### flat seed 19 — latent dimension z2

Validation R² for this latent: **0.826**

1. Formula: `mean(x[7,112:128])`
   - Raw location: `total_acc_y[112:128]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_y local mean level**
   - Lasso coefficient: `-4.8765`

2. Formula: `mean(square(x[0,89:105]))`
   - Raw location: `body_acc_x[89:105]` (0.32 s window)
   - Automatic structural interpretation: **body_acc_x local signal magnitude**
   - Lasso coefficient: `-3.3280`

3. Formula: `mean(square(x[8,67:83]))`
   - Raw location: `total_acc_z[67:83]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_z local signal magnitude**
   - Lasso coefficient: `-2.8994`

### flat seed 19 — latent dimension z3

Validation R² for this latent: **0.731**

1. Formula: `mean(x[6,29:45])`
   - Raw location: `total_acc_x[29:45]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_x local mean level**
   - Lasso coefficient: `-5.1854`

2. Formula: `mean(square(x[0,18:34]))`
   - Raw location: `body_acc_x[18:34]` (0.32 s window)
   - Automatic structural interpretation: **body_acc_x local signal magnitude**
   - Lasso coefficient: `-3.9688`

3. Formula: `mean(abs(diff(x[4,45:61])))`
   - Raw location: `body_gyro_y[45:61]` (0.32 s window)
   - Automatic structural interpretation: **body_gyro_y local temporal change**
   - Lasso coefficient: `-2.4671`

### flat seed 19 — latent dimension z4

Validation R² for this latent: **0.912**

1. Formula: `mean(square(x[6,44:60]))`
   - Raw location: `total_acc_x[44:60]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_x local signal magnitude**
   - Lasso coefficient: `4.4197`

2. Formula: `mean(x[6,44:60])`
   - Raw location: `total_acc_x[44:60]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_x local mean level**
   - Lasso coefficient: `-2.4418`

3. Formula: `mean(x[7,19:35])`
   - Raw location: `total_acc_y[19:35]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_y local mean level**
   - Lasso coefficient: `1.1706`

### flat seed 23 — latent dimension z1

Validation R² for this latent: **0.861**

1. Formula: `mean(x[6,91:107])`
   - Raw location: `total_acc_x[91:107]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_x local mean level**
   - Lasso coefficient: `-4.3694`

2. Formula: `mean(square(x[6,91:107]))`
   - Raw location: `total_acc_x[91:107]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_x local signal magnitude**
   - Lasso coefficient: `3.0955`

3. Formula: `mean(square(x[7,66:82]))`
   - Raw location: `total_acc_y[66:82]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_y local signal magnitude**
   - Lasso coefficient: `1.6740`

### flat seed 23 — latent dimension z2

Validation R² for this latent: **0.789**

1. Formula: `mean(x[7,7:23])`
   - Raw location: `total_acc_y[7:23]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_y local mean level**
   - Lasso coefficient: `3.4419`

2. Formula: `mean(square(x[7,7:23]))`
   - Raw location: `total_acc_y[7:23]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_y local signal magnitude**
   - Lasso coefficient: `-2.5916`

3. Formula: `mean(abs(diff(x[4,77:93])))`
   - Raw location: `body_gyro_y[77:93]` (0.32 s window)
   - Automatic structural interpretation: **body_gyro_y local temporal change**
   - Lasso coefficient: `2.0531`

### flat seed 23 — latent dimension z3

Validation R² for this latent: **0.658**

1. Formula: `mean(x[7,112:128])`
   - Raw location: `total_acc_y[112:128]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_y local mean level**
   - Lasso coefficient: `3.8203`

2. Formula: `mean(square(x[7,112:128]))`
   - Raw location: `total_acc_y[112:128]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_y local signal magnitude**
   - Lasso coefficient: `-2.9795`

3. Formula: `max(x[5,28:44])-min(x[5,28:44])`
   - Raw location: `body_gyro_z[28:44]` (0.32 s window)
   - Automatic structural interpretation: **body_gyro_z local movement range**
   - Lasso coefficient: `1.9796`

### flat seed 23 — latent dimension z4

Validation R² for this latent: **0.865**

1. Formula: `mean(x[6,67:83])`
   - Raw location: `total_acc_x[67:83]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_x local mean level**
   - Lasso coefficient: `-6.5178`

2. Formula: `mean(square(x[0,43:59]))`
   - Raw location: `body_acc_x[43:59]` (0.32 s window)
   - Automatic structural interpretation: **body_acc_x local signal magnitude**
   - Lasso coefficient: `-2.3074`

3. Formula: `max(x[7,109:125])-min(x[7,109:125])`
   - Raw location: `total_acc_y[109:125]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_y local movement range**
   - Lasso coefficient: `1.3740`

### flat seed 31 — latent dimension z1

Validation R² for this latent: **0.950**

1. Formula: `mean(x[7,16:32])`
   - Raw location: `total_acc_y[16:32]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_y local mean level**
   - Lasso coefficient: `-5.3241`

2. Formula: `mean(square(x[8,88:104]))`
   - Raw location: `total_acc_z[88:104]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_z local signal magnitude**
   - Lasso coefficient: `-3.3262`

3. Formula: `mean(x[8,88:104])`
   - Raw location: `total_acc_z[88:104]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_z local mean level**
   - Lasso coefficient: `-2.8910`

### flat seed 31 — latent dimension z2

Validation R² for this latent: **0.574**

1. Formula: `mean(x[7,41:57])`
   - Raw location: `total_acc_y[41:57]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_y local mean level**
   - Lasso coefficient: `3.0527`

2. Formula: `mean(x[7,41:53]*x[6,41:53])`
   - Raw location: `total_acc_y[41:53]` (0.24 s window) and `total_acc_x[41:53]` (0.24 s window)
   - Automatic structural interpretation: **joint signal strength of total_acc_y and total_acc_x**
   - Lasso coefficient: `2.0648`

3. Formula: `mean(x[6,37:53])`
   - Raw location: `total_acc_x[37:53]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_x local mean level**
   - Lasso coefficient: `1.2588`

### flat seed 31 — latent dimension z3

Validation R² for this latent: **0.868**

1. Formula: `mean(x[7,91:107])`
   - Raw location: `total_acc_y[91:107]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_y local mean level**
   - Lasso coefficient: `-2.8027`

2. Formula: `max(x[5,84:100])-min(x[5,84:100])`
   - Raw location: `body_gyro_z[84:100]` (0.32 s window)
   - Automatic structural interpretation: **body_gyro_z local movement range**
   - Lasso coefficient: `2.6029`

3. Formula: `max(x[0,54:70])-min(x[0,54:70])`
   - Raw location: `body_acc_x[54:70]` (0.32 s window)
   - Automatic structural interpretation: **body_acc_x local movement range**
   - Lasso coefficient: `2.3314`

### flat seed 31 — latent dimension z4

Validation R² for this latent: **0.941**

1. Formula: `mean(x[6,28:44])`
   - Raw location: `total_acc_x[28:44]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_x local mean level**
   - Lasso coefficient: `4.4927`

2. Formula: `mean(x[6,28:41]*x[7,28:41])`
   - Raw location: `total_acc_x[28:41]` (0.26 s window) and `total_acc_y[28:41]` (0.26 s window)
   - Automatic structural interpretation: **joint signal strength of total_acc_x and total_acc_y**
   - Lasso coefficient: `3.3479`

3. Formula: `max(x[0,39:55])-min(x[0,39:55])`
   - Raw location: `body_acc_x[39:55]` (0.32 s window)
   - Automatic structural interpretation: **body_acc_x local movement range**
   - Lasso coefficient: `-2.1012`

### wide seed 7 — latent dimension z1

Validation R² for this latent: **0.643**

1. Formula: `mean(x[6,80:96])`
   - Raw location: `total_acc_x[80:96]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_x local mean level**
   - Lasso coefficient: `-1.7927`

2. Formula: `mean(square(x[0,42:58]))`
   - Raw location: `body_acc_x[42:58]` (0.32 s window)
   - Automatic structural interpretation: **body_acc_x local signal magnitude**
   - Lasso coefficient: `-1.4359`

3. Formula: `max(x[0,42:58])-min(x[0,42:58])`
   - Raw location: `body_acc_x[42:58]` (0.32 s window)
   - Automatic structural interpretation: **body_acc_x local movement range**
   - Lasso coefficient: `1.0839`

### wide seed 7 — latent dimension z2

Validation R² for this latent: **0.953**

1. Formula: `mean(x[7,10:26])`
   - Raw location: `total_acc_y[10:26]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_y local mean level**
   - Lasso coefficient: `-8.3706`

2. Formula: `mean(x[8,110:126])`
   - Raw location: `total_acc_z[110:126]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_z local mean level**
   - Lasso coefficient: `-3.7829`

3. Formula: `mean(square(x[8,110:126]))`
   - Raw location: `total_acc_z[110:126]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_z local signal magnitude**
   - Lasso coefficient: `-2.0385`

### wide seed 7 — latent dimension z3

Validation R² for this latent: **0.867**

1. Formula: `mean(x[6,29:45])`
   - Raw location: `total_acc_x[29:45]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_x local mean level**
   - Lasso coefficient: `-8.9362`

2. Formula: `max(x[6,29:45])-min(x[6,29:45])`
   - Raw location: `total_acc_x[29:45]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_x local movement range**
   - Lasso coefficient: `3.1658`

3. Formula: `max(x[0,34:50])-min(x[0,34:50])`
   - Raw location: `body_acc_x[34:50]` (0.32 s window)
   - Automatic structural interpretation: **body_acc_x local movement range**
   - Lasso coefficient: `1.9554`

### wide seed 7 — latent dimension z4

Validation R² for this latent: **0.663**

1. Formula: `mean(x[7,64:80])`
   - Raw location: `total_acc_y[64:80]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_y local mean level**
   - Lasso coefficient: `2.8255`

2. Formula: `mean(square(x[0,88:104]))`
   - Raw location: `body_acc_x[88:104]` (0.32 s window)
   - Automatic structural interpretation: **body_acc_x local signal magnitude**
   - Lasso coefficient: `1.9132`

3. Formula: `mean(x[8,60:76])`
   - Raw location: `total_acc_z[60:76]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_z local mean level**
   - Lasso coefficient: `1.9105`

### wide seed 11 — latent dimension z1

Validation R² for this latent: **0.737**

1. Formula: `mean(x[6,98:114])`
   - Raw location: `total_acc_x[98:114]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_x local mean level**
   - Lasso coefficient: `-4.3223`

2. Formula: `mean(square(x[0,106:122]))`
   - Raw location: `body_acc_x[106:122]` (0.32 s window)
   - Automatic structural interpretation: **body_acc_x local signal magnitude**
   - Lasso coefficient: `-2.8943`

3. Formula: `mean(abs(diff(x[4,30:46])))`
   - Raw location: `body_gyro_y[30:46]` (0.32 s window)
   - Automatic structural interpretation: **body_gyro_y local temporal change**
   - Lasso coefficient: `-2.2345`

### wide seed 11 — latent dimension z2

Validation R² for this latent: **0.728**

1. Formula: `mean(x[6,96:112])`
   - Raw location: `total_acc_x[96:112]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_x local mean level**
   - Lasso coefficient: `-2.3678`

2. Formula: `mean(square(x[6,96:112]))`
   - Raw location: `total_acc_x[96:112]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_x local signal magnitude**
   - Lasso coefficient: `1.8224`

3. Formula: `max(x[6,96:112])-min(x[6,96:112])`
   - Raw location: `total_acc_x[96:112]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_x local movement range**
   - Lasso coefficient: `-1.1918`

### wide seed 11 — latent dimension z3

Validation R² for this latent: **0.953**

1. Formula: `mean(x[6,96:112])`
   - Raw location: `total_acc_x[96:112]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_x local mean level**
   - Lasso coefficient: `3.9758`

2. Formula: `mean(x[6,96:105]*x[7,96:105])`
   - Raw location: `total_acc_x[96:105]` (0.18 s window) and `total_acc_y[96:105]` (0.18 s window)
   - Automatic structural interpretation: **joint signal strength of total_acc_x and total_acc_y**
   - Lasso coefficient: `2.4354`

3. Formula: `mean(x[8,77:93])`
   - Raw location: `total_acc_z[77:93]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_z local mean level**
   - Lasso coefficient: `-2.1976`

### wide seed 11 — latent dimension z4

Validation R² for this latent: **0.710**

1. Formula: `mean(x[7,75:91])`
   - Raw location: `total_acc_y[75:91]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_y local mean level**
   - Lasso coefficient: `4.7001`

2. Formula: `mean(x[6,96:112])`
   - Raw location: `total_acc_x[96:112]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_x local mean level**
   - Lasso coefficient: `3.0117`

3. Formula: `mean(square(x[6,96:112]))`
   - Raw location: `total_acc_x[96:112]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_x local signal magnitude**
   - Lasso coefficient: `-2.4180`

### wide seed 19 — latent dimension z1

Validation R² for this latent: **0.754**

1. Formula: `mean(square(x[6,48:64]))`
   - Raw location: `total_acc_x[48:64]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_x local signal magnitude**
   - Lasso coefficient: `-2.9852`

2. Formula: `mean(x[7,92:108])`
   - Raw location: `total_acc_y[92:108]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_y local mean level**
   - Lasso coefficient: `2.4933`

3. Formula: `mean(abs(diff(x[4,57:73])))`
   - Raw location: `body_gyro_y[57:73]` (0.32 s window)
   - Automatic structural interpretation: **body_gyro_y local temporal change**
   - Lasso coefficient: `-2.1078`

### wide seed 19 — latent dimension z2

Validation R² for this latent: **0.944**

1. Formula: `mean(x[7,1:17])`
   - Raw location: `total_acc_y[1:17]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_y local mean level**
   - Lasso coefficient: `5.2983`

2. Formula: `mean(square(x[7,1:17]))`
   - Raw location: `total_acc_y[1:17]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_y local signal magnitude**
   - Lasso coefficient: `1.5747`

3. Formula: `max(x[6,46:62])-min(x[6,46:62])`
   - Raw location: `total_acc_x[46:62]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_x local movement range**
   - Lasso coefficient: `-1.2995`

### wide seed 19 — latent dimension z3

Validation R² for this latent: **0.531**

1. Formula: `mean(x[7,15:31])`
   - Raw location: `total_acc_y[15:31]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_y local mean level**
   - Lasso coefficient: `-2.4001`

2. Formula: `mean(x[6,103:112]*x[8,103:112])`
   - Raw location: `total_acc_x[103:112]` (0.18 s window) and `total_acc_z[103:112]` (0.18 s window)
   - Automatic structural interpretation: **joint signal strength of total_acc_x and total_acc_z**
   - Lasso coefficient: `-1.8403`

3. Formula: `mean(x[8,103:119])`
   - Raw location: `total_acc_z[103:119]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_z local mean level**
   - Lasso coefficient: `-1.5270`

### wide seed 19 — latent dimension z4

Validation R² for this latent: **0.858**

1. Formula: `mean(x[6,92:108])`
   - Raw location: `total_acc_x[92:108]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_x local mean level**
   - Lasso coefficient: `-5.5322`

2. Formula: `mean(square(x[6,92:108]))`
   - Raw location: `total_acc_x[92:108]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_x local signal magnitude**
   - Lasso coefficient: `4.3946`

3. Formula: `mean(square(x[0,67:83]))`
   - Raw location: `body_acc_x[67:83]` (0.32 s window)
   - Automatic structural interpretation: **body_acc_x local signal magnitude**
   - Lasso coefficient: `-3.2213`

### wide seed 23 — latent dimension z1

Validation R² for this latent: **0.750**

1. Formula: `mean(square(x[0,33:49]))`
   - Raw location: `body_acc_x[33:49]` (0.32 s window)
   - Automatic structural interpretation: **body_acc_x local signal magnitude**
   - Lasso coefficient: `-2.3746`

2. Formula: `mean(square(x[6,95:111]))`
   - Raw location: `total_acc_x[95:111]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_x local signal magnitude**
   - Lasso coefficient: `-1.9632`

3. Formula: `max(x[0,33:49])-min(x[0,33:49])`
   - Raw location: `body_acc_x[33:49]` (0.32 s window)
   - Automatic structural interpretation: **body_acc_x local movement range**
   - Lasso coefficient: `1.7624`

### wide seed 23 — latent dimension z2

Validation R² for this latent: **0.677**

1. Formula: `max(x[6,13:29])-min(x[6,13:29])`
   - Raw location: `total_acc_x[13:29]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_x local movement range**
   - Lasso coefficient: `-3.6987`

2. Formula: `max(x[0,38:54])-min(x[0,38:54])`
   - Raw location: `body_acc_x[38:54]` (0.32 s window)
   - Automatic structural interpretation: **body_acc_x local movement range**
   - Lasso coefficient: `-2.4170`

3. Formula: `mean(x[6,13:29])`
   - Raw location: `total_acc_x[13:29]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_x local mean level**
   - Lasso coefficient: `1.7110`

### wide seed 23 — latent dimension z3

Validation R² for this latent: **0.743**

1. Formula: `mean(x[6,15:31])`
   - Raw location: `total_acc_x[15:31]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_x local mean level**
   - Lasso coefficient: `-6.2245`

2. Formula: `max(x[0,48:64])-min(x[0,48:64])`
   - Raw location: `body_acc_x[48:64]` (0.32 s window)
   - Automatic structural interpretation: **body_acc_x local movement range**
   - Lasso coefficient: `4.8810`

3. Formula: `mean(square(x[0,48:64]))`
   - Raw location: `body_acc_x[48:64]` (0.32 s window)
   - Automatic structural interpretation: **body_acc_x local signal magnitude**
   - Lasso coefficient: `-3.7025`

### wide seed 23 — latent dimension z4

Validation R² for this latent: **0.771**

1. Formula: `mean(x[7,66:82])`
   - Raw location: `total_acc_y[66:82]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_y local mean level**
   - Lasso coefficient: `-3.6273`

2. Formula: `mean(square(x[0,39:55]))`
   - Raw location: `body_acc_x[39:55]` (0.32 s window)
   - Automatic structural interpretation: **body_acc_x local signal magnitude**
   - Lasso coefficient: `-2.8077`

3. Formula: `mean(abs(diff(x[8,5:21])))`
   - Raw location: `total_acc_z[5:21]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_z local temporal change**
   - Lasso coefficient: `-2.7244`

### wide seed 31 — latent dimension z1

Validation R² for this latent: **0.732**

1. Formula: `max(x[2,0:16])-min(x[2,0:16])`
   - Raw location: `body_acc_z[0:16]` (0.32 s window)
   - Automatic structural interpretation: **body_acc_z local movement range**
   - Lasso coefficient: `2.6113`

2. Formula: `max(x[0,44:60])-min(x[0,44:60])`
   - Raw location: `body_acc_x[44:60]` (0.32 s window)
   - Automatic structural interpretation: **body_acc_x local movement range**
   - Lasso coefficient: `2.4780`

3. Formula: `mean(x[6,105:121])`
   - Raw location: `total_acc_x[105:121]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_x local mean level**
   - Lasso coefficient: `-2.1472`

### wide seed 31 — latent dimension z2

Validation R² for this latent: **0.681**

1. Formula: `mean(x[7,26:42])`
   - Raw location: `total_acc_y[26:42]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_y local mean level**
   - Lasso coefficient: `3.5456`

2. Formula: `mean(square(x[0,58:74]))`
   - Raw location: `body_acc_x[58:74]` (0.32 s window)
   - Automatic structural interpretation: **body_acc_x local signal magnitude**
   - Lasso coefficient: `2.5128`

3. Formula: `max(x[6,26:42])-min(x[6,26:42])`
   - Raw location: `total_acc_x[26:42]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_x local movement range**
   - Lasso coefficient: `1.8511`

### wide seed 31 — latent dimension z3

Validation R² for this latent: **0.926**

1. Formula: `mean(square(x[6,110:126]))`
   - Raw location: `total_acc_x[110:126]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_x local signal magnitude**
   - Lasso coefficient: `3.1980`

2. Formula: `mean(x[6,110:126])`
   - Raw location: `total_acc_x[110:126]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_x local mean level**
   - Lasso coefficient: `-2.7909`

3. Formula: `max(x[0,45:61])-min(x[0,45:61])`
   - Raw location: `body_acc_x[45:61]` (0.32 s window)
   - Automatic structural interpretation: **body_acc_x local movement range**
   - Lasso coefficient: `-2.2829`

### wide seed 31 — latent dimension z4

Validation R² for this latent: **0.853**

1. Formula: `mean(x[6,87:103])`
   - Raw location: `total_acc_x[87:103]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_x local mean level**
   - Lasso coefficient: `6.4589`

2. Formula: `mean(x[7,112:128])`
   - Raw location: `total_acc_y[112:128]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_y local mean level**
   - Lasso coefficient: `4.6103`

3. Formula: `mean(square(x[6,87:103]))`
   - Raw location: `total_acc_x[87:103]` (0.32 s window)
   - Automatic structural interpretation: **total_acc_x local signal magnitude**
   - Lasso coefficient: `-3.1268`

## What the person can conclude

The AI repeatedly found short raw channel-time regions and simple local operations that predict parts of the learned four-dimensional representation. The most common human-readable interpretations are local signal magnitude, local range, local temporal change, and local mean level. These are interpretations of the discovered formulas, not feature names supplied during training.

## Boundary

A latent dimension is not a universal physical variable: latent coordinates can rotate between models, and the subspace similarity is moderate. The report therefore treats recurring formula structures and raw regions as candidates for measurement definitions, not as proven causes of the activity distinction.
