# W1 우세의 적용 범위

다른 표준 dataset은 sklearn이 설치되지 않아 실행하지 않았다.

## adam_baseline

`{"layer1": {"parameter_update_norm": 0.11570514471527175, "jacobian_amplification": 8.008795618446127, "same_norm_geometry_gain": 4.090195066407682, "geometry_gain_per_parameter_norm": 3.1218782161889345, "geometry_gain_per_hidden_norm": 0.40618673915580833}, "layer2": {"parameter_update_norm": 0.03032634983308973, "jacobian_amplification": 3.057631688628725, "same_norm_geometry_gain": 0.8497945294924577, "geometry_gain_per_parameter_norm": 0.6500591306574611, "geometry_gain_per_hidden_norm": 0.20637767699401607}}`

## sgd_baseline

`{"layer1": {"parameter_update_norm": 0.00536674040438436, "jacobian_amplification": 11.97433868553656, "same_norm_geometry_gain": 4.5387662461565865, "geometry_gain_per_parameter_norm": -0.5445247999896177, "geometry_gain_per_hidden_norm": -0.041479854808155}, "layer2": {"parameter_update_norm": 0.002113950323949043, "jacobian_amplification": 3.983347918968809, "same_norm_geometry_gain": 0.514262313613012, "geometry_gain_per_parameter_norm": -0.15016881570818214, "geometry_gain_per_hidden_norm": -0.03758611621132448}}`

## deep

`{"layer1": {"parameter_update_norm": 0.11397553728898976, "jacobian_amplification": 8.27389818945911, "same_norm_geometry_gain": 3.4739016782874286, "geometry_gain_per_parameter_norm": 2.5974128838851236, "geometry_gain_per_hidden_norm": 0.3314305498803202}, "layer2": {"parameter_update_norm": 0.04116019450436495, "jacobian_amplification": 2.970713477670994, "same_norm_geometry_gain": 0.7006352753308346, "geometry_gain_per_parameter_norm": 0.5595833746143637, "geometry_gain_per_hidden_norm": 0.18691062351201698}, "layer3": {"parameter_update_norm": 0.029643389011345042, "jacobian_amplification": 3.1880052040809432, "same_norm_geometry_gain": 0.4391245840327386, "geometry_gain_per_parameter_norm": 0.3039979190316272, "geometry_gain_per_hidden_norm": 0.0935371139747386}}`

## narrow

`{"layer1": {"parameter_update_norm": 0.08649066503582645, "jacobian_amplification": 7.982133178029885, "same_norm_geometry_gain": 4.122365974578439, "geometry_gain_per_parameter_norm": 2.4981851587200183, "geometry_gain_per_hidden_norm": 0.3143878482643169}, "layer2": {"parameter_update_norm": 0.015942540546288374, "jacobian_amplification": 2.24635826035031, "same_norm_geometry_gain": 0.32563354061528993, "geometry_gain_per_parameter_norm": 0.20102523605542175, "geometry_gain_per_hidden_norm": 0.07860729235643278}}`

## wide

`{"layer1": {"parameter_update_norm": 0.15604479822707226, "jacobian_amplification": 7.899503251684546, "same_norm_geometry_gain": 3.6910616682000197, "geometry_gain_per_parameter_norm": 2.9143165204651122, "geometry_gain_per_hidden_norm": 0.38333000011046187}, "layer2": {"parameter_update_norm": 0.055577185270345136, "jacobian_amplification": 3.7995984089480697, "same_norm_geometry_gain": 1.0091292220482178, "geometry_gain_per_parameter_norm": 0.7814453633527925, "geometry_gain_per_hidden_norm": 0.20126701917345197}}`

## pca_whiten

`{"layer1": {"parameter_update_norm": 0.05261167773632582, "jacobian_amplification": 1.1196524158337064, "same_norm_geometry_gain": 0.05534128832598559, "geometry_gain_per_parameter_norm": 0.00016888451119941806, "geometry_gain_per_hidden_norm": 0.002690413776069565}, "layer2": {"parameter_update_norm": 0.03380776845675648, "jacobian_amplification": 3.0268177571908272, "same_norm_geometry_gain": -0.034222959403315684, "geometry_gain_per_parameter_norm": -0.06750747613646492, "geometry_gain_per_hidden_norm": -0.02194381855671652}}`

## 최소 해석

입력에 가까운 층의 우세가 optimizer, architecture, input geometry에 따라 유지되는지 summary와 원자료를 비교한다.
