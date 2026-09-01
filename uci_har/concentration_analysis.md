# Input class-information concentration과 Adam geometry

## 1

- 가설: decorrelation은 class information을 소수 coordinate에 집중시킨다.
- 통제 실험: scale-only/decorrelated/random rotation/PCA information spread의 top-share·effective count를 비교한다.
- 실제 결과: `{"scale_only": {"top10_share": 0.18636482256264283, "effective_coordinate_count": 409.63459530333125, "herfindahl": 0.002441200063338175}, "decorrelated": {"top10_share": 0.9991922317465248, "effective_coordinate_count": 1.240824193085292, "herfindahl": 0.8059159432679291}, "random_rotated": {"top10_share": 0.4049201871447956, "effective_coordinate_count": 220.5883165378662, "herfindahl": 0.004533331663684644}, "pca_info_spread": {"top10_share": 0.13858524121655152, "effective_coordinate_count": 519.2638996997104, "herfindahl": 0.0019258030465401095}}`
- 모순: concentration이 gradient와 v에 전달되는 단계별 차이를 분리한다.
- 수정된 원리: input basis/eigenstructure가 class information을 집중시키고, 그 집중이 gradient와 v imbalance를 거쳐 geometry를 약화한다.

## 2

- 가설: 집중된 input class information이 gradient concentration을 만든다.
- 통제 실험: class-information top coordinate와 W1 gradient/class contribution을 비교한다.
- 실제 결과: `{"scale_only": {"gradient": {"top10_share": 0.11577839529766791, "effective_coordinate_count": 507.78009407640747, "herfindahl": 0.00197120104031883}, "adam_alignment": 0.8846722438187494}, "decorrelated": {"gradient": {"top10_share": 0.609018899412377, "effective_coordinate_count": 27.842461920711344, "herfindahl": 0.03799825897016461}, "adam_alignment": 0.2635861033347211}, "random_rotated": {"gradient": {"top10_share": 0.18823290139334495, "effective_coordinate_count": 468.72936908280843, "herfindahl": 0.002141400047003035}, "adam_alignment": 0.8192179313232077}, "pca_info_spread": {"gradient": {"top10_share": 0.06731402382022954, "effective_coordinate_count": 315.762722584428, "herfindahl": 0.0031694698595723016}, "adam_alignment": 0.25874373434355175}}`
- 모순: concentration이 gradient와 v에 전달되는 단계별 차이를 분리한다.
- 수정된 원리: input basis/eigenstructure가 class information을 집중시키고, 그 집중이 gradient와 v imbalance를 거쳐 geometry를 약화한다.

## 3

- 가설: gradient concentration이 v_t concentration으로 이어진다.
- 통제 실험: coordinate별 class-info→gradient→v 상관과 top-v share를 추적한다.
- 실제 결과: `{"scale_only": {"class_info_v": 0.8506852891512282, "gradient_v": 0.6147501459164032, "v_concentration": {"top10_share": 0.15496977750573845, "effective_coordinate_count": 453.90146518202795, "herfindahl": 0.002206678442211248}}, "decorrelated": {"class_info_v": 0.9997918535884095, "gradient_v": 0.9258320835591118, "v_concentration": {"top10_share": 0.9810918207722997, "effective_coordinate_count": 1.418311923567102, "herfindahl": 0.7080550025287198}}, "random_rotated": {"class_info_v": 0.9720615896119207, "gradient_v": 0.8502617745857174, "v_concentration": {"top10_share": 0.3517456093537929, "effective_coordinate_count": 265.39455542099086, "herfindahl": 0.0038023220850962438}}, "pca_info_spread": {"class_info_v": -0.16892487971252787, "gradient_v": 0.8664397011738025, "v_concentration": {"top10_share": 0.028925100318793202, "effective_coordinate_count": 114.20883795126963, "herfindahl": 0.008854429067086673}}}`
- 모순: concentration이 gradient와 v에 전달되는 단계별 차이를 분리한다.
- 수정된 원리: input basis/eigenstructure가 class information을 집중시키고, 그 집중이 gradient와 v imbalance를 거쳐 geometry를 약화한다.

## 4

- 가설: concentration 하나가 Adam alignment와 geometry gain을 예측한다.
- 통제 실험: 조건·seed aggregate에서 input top-share와 Adam alignment/gain을 비교한다.
- 실제 결과: `{"scale_only": {"input_top10_share": 0.18636482256264283, "adam_alignment": 0.8846722438187494, "adam_geometry_gain": 4.0927265470398195}, "decorrelated": {"input_top10_share": 0.9991922317465248, "adam_alignment": 0.2635861033347211, "adam_geometry_gain": 0.36392321351424967}, "random_rotated": {"input_top10_share": 0.4049201871447956, "adam_alignment": 0.8192179313232077, "adam_geometry_gain": 3.4297248626132624}, "pca_info_spread": {"input_top10_share": 0.13858524121655152, "adam_alignment": 0.25874373434355175, "adam_geometry_gain": 0.002260739763551125}}`
- 모순: concentration이 gradient와 v에 전달되는 단계별 차이를 분리한다.
- 수정된 원리: input basis/eigenstructure가 class information을 집중시키고, 그 집중이 gradient와 v imbalance를 거쳐 geometry를 약화한다.

## 최소 원리

`input basis/eigenstructure → class information concentration → gradient concentration → v_t imbalance → class-important coordinate suppression → hidden geometry 감소`
