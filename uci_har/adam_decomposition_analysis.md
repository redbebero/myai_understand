# Adam 내부 연산과 W1 class-direction 정렬

## 1

- 수식의 예측: input geometry가 raw gradient의 class-direction 정렬을 만든다.
- 통제 실험: scale-only와 decorrelated 입력에서 g_t를 비교한다.
- 실제 결과: `{"scale_only": {"raw_norm": 2.6997273788950364, "alignment": 0.8933221260890971, "same_norm_geometry_gain": 3.452860009167532, "hidden_class_distance_gap_change": 3.452860009167532}, "decorrelated": {"raw_norm": 4.663401256121993, "alignment": 0.9536980868642368, "same_norm_geometry_gain": 5.13472869047843, "hidden_class_distance_gap_change": 5.134728690478431}}`
- 맞지 않는 점: scale-only와 decorrelated 사이에서 alignment·geometry gain이 달라지는 단계를 확인한다.
- 수정된 원리: Adam의 특정 연산이 입력 class-direction과 coordinate-wise update를 재가중한다.

## 2

- 수식의 예측: momentum은 gradient 정렬을 유지한다.
- 통제 실험: m_t만 같은 norm으로 적용한다.
- 실제 결과: `{"scale_only": {"raw_norm": 2.7632058692785493, "alignment": 0.9620235813185789, "same_norm_geometry_gain": 4.755992482610674, "hidden_class_distance_gap_change": 4.755992482610674}, "decorrelated": {"raw_norm": 4.967970602198861, "alignment": 0.9833571266033558, "same_norm_geometry_gain": 5.347439671569945, "hidden_class_distance_gap_change": 5.347439671569945}}`
- 맞지 않는 점: scale-only와 decorrelated 사이에서 alignment·geometry gain이 달라지는 단계를 확인한다.
- 수정된 원리: Adam의 특정 연산이 입력 class-direction과 coordinate-wise update를 재가중한다.

## 3

- 수식의 예측: second-moment scaling은 좌표별 variance를 보정하지만 방향을 크게 바꾸지 않는다.
- 통제 실험: g_t/√v_t만 같은 norm으로 적용한다.
- 실제 결과: `{"scale_only": {"raw_norm": 145.8557139469911, "alignment": 0.7978551295757196, "same_norm_geometry_gain": 2.6766468943239334, "hidden_class_distance_gap_change": 2.6766468943239334}, "decorrelated": {"raw_norm": 164.9020220803429, "alignment": 0.14538185434255782, "same_norm_geometry_gain": 0.27359328520166576, "hidden_class_distance_gap_change": 0.27359328520166565}}`
- 맞지 않는 점: scale-only와 decorrelated 사이에서 alignment·geometry gain이 달라지는 단계를 확인한다.
- 수정된 원리: Adam의 특정 연산이 입력 class-direction과 coordinate-wise update를 재가중한다.

## 4

- 수식의 예측: full Adam은 앞 단계의 장점을 결합한다.
- 통제 실험: m_t/√v_t를 same-norm으로 적용하고 각 단계와 비교한다.
- 실제 결과: `{"scale_only": {"raw_norm": 115.58473214054264, "alignment": 0.8846722438187494, "same_norm_geometry_gain": 4.0927265470398195, "hidden_class_distance_gap_change": 4.0927265470398195}, "decorrelated": {"raw_norm": 92.491786912858, "alignment": 0.2635861033347211, "same_norm_geometry_gain": 0.36392321351424967, "hidden_class_distance_gap_change": 0.36392321351424967}}`
- 맞지 않는 점: scale-only와 decorrelated 사이에서 alignment·geometry gain이 달라지는 단계를 확인한다.
- 수정된 원리: Adam의 특정 연산이 입력 class-direction과 coordinate-wise update를 재가중한다.

## 최소 메커니즘

`input geometry → g_t → m_t → 1/√v_t coordinate reweighting → actual Adam update → hidden geometry`
