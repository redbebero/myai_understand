# Adam second-moment scaling과 class-direction 손실

## 1

- 가설: decorrelated 입력은 class-important gradient coordinate에 더 큰 v_t를 만든다.
- 통제 실험: scale-only와 decorrelated의 v_t 분포 및 top class-coordinate를 비교한다.
- 실제 결과: `{"scale_only": {"coefficient_variation": 1.5086593203744254, "class_contribution_v_correlation": 0.8279306623984415, "class_contribution_scaling_correlation": -0.8102070161042062}, "decorrelated": {"coefficient_variation": 33.14525685154249, "class_contribution_v_correlation": 0.9538922927647867, "class_contribution_scaling_correlation": -0.010083102277068857}}`
- 맞지 않는 점: v 크기·coordinate 불균형·class-coordinate 억제를 분리해 확인한다.
- 수정된 원리: second-moment scaling이 class-aligned gradient를 coordinate-wise로 재가중하는 단계로 설명한다.

## 2

- 가설: 큰 v_t가 class-important coordinate를 선택적으로 축소한다.
- 통제 실험: top class-contribution coordinate와 나머지의 v_t·scaling factor를 비교한다.
- 실제 결과: `{"scale_only": {"class_contribution": [0.15494094609426282, 0.08961871224775772], "gradient_magnitude": [0.1556646128922675, 0.10307138355562112], "v": [0.0007667463748865646, 0.0004426931516425309], "scaling_factor": [49.79196283846741, 64.11850715872761], "effective_momentum_scale": [39.39750120135363, 50.30105753810222]}, "decorrelated": {"class_contribution": [0.1822838873285655, 0.0032015089630659938], "gradient_magnitude": [0.25454297668218767, 0.01855431510353323], "v": [0.008108204842614685, 1.740397676468815e-05], "scaling_factor": [58.80870576196401, 657.2827547781826], "effective_momentum_scale": [53.89430639145682, 641.5578339538218]}}`
- 맞지 않는 점: v 크기·coordinate 불균형·class-coordinate 억제를 분리해 확인한다.
- 수정된 원리: second-moment scaling이 class-aligned gradient를 coordinate-wise로 재가중하는 단계로 설명한다.

## 3

- 가설: coordinate-wise scaling이 class direction을 보존하지 못한다.
- 통제 실험: momentum, full Adam, scalar-v, clipped-v의 alignment와 same-norm gain을 비교한다.
- 실제 결과: `{"scale_only": {"momentum": {"alignment": 0.9620235813185789, "same_norm_geometry_gain": 4.755992482610674, "hidden_class_distance_gap_change": 4.755992482610674}, "full_adam": {"alignment": 0.8846722438187494, "same_norm_geometry_gain": 4.0927265470398195, "hidden_class_distance_gap_change": 4.0927265470398195}, "scalar_v": {"alignment": 0.9620235813185795, "same_norm_geometry_gain": 4.755992482610676, "hidden_class_distance_gap_change": 4.755992482610674}, "clipped_v": {"alignment": 0.9082486313631587, "same_norm_geometry_gain": 4.34117904742853, "hidden_class_distance_gap_change": 4.34117904742853}}, "decorrelated": {"momentum": {"alignment": 0.9833571266033558, "same_norm_geometry_gain": 5.347439671569945, "hidden_class_distance_gap_change": 5.347439671569945}, "full_adam": {"alignment": 0.2635861033347211, "same_norm_geometry_gain": 0.36392321351424967, "hidden_class_distance_gap_change": 0.36392321351424967}, "scalar_v": {"alignment": 0.9833571266033556, "same_norm_geometry_gain": 5.347439671569942, "hidden_class_distance_gap_change": 5.347439671569942}, "clipped_v": {"alignment": 0.9693308704628045, "same_norm_geometry_gain": 5.249918190627488, "hidden_class_distance_gap_change": 5.24991819062749}}}`
- 맞지 않는 점: v 크기·coordinate 불균형·class-coordinate 억제를 분리해 확인한다.
- 수정된 원리: second-moment scaling이 class-aligned gradient를 coordinate-wise로 재가중하는 단계로 설명한다.

## 최소 메커니즘

`input geometry → gradient coordinate structure → v_t → 1/√v_t scaling → class-direction loss → hidden geometry 감소`
