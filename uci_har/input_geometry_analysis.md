# 입력 geometry와 W1 우세 분해

## 1

- 가설: feature scale만 정규화해도 W1 geometry 형성이 유지된다.
- 통제 실험: unscaled vs scale_only
- 실제 결과: `{"unscaled": {"parameter_update_norm": 0.10413756028253056, "jacobian_amplification": 7.055817903586693, "same_norm_geometry_gain": 1.7388700769602823, "hidden_class_distance_gap_change": 0.14896613037593995, "geometry_gain_per_hidden_norm": 0.2724948175688706, "input_gradient_alignment": 0.7401070128187024, "input_update_alignment": 0.675409076118939}, "scale_only": {"parameter_update_norm": 0.11570514471527175, "jacobian_amplification": 8.008795618446127, "same_norm_geometry_gain": 4.090195066407682, "hidden_class_distance_gap_change": 0.33271800264641127, "geometry_gain_per_hidden_norm": 0.40618673915580833, "input_gradient_alignment": 0.8933221260890971, "input_update_alignment": 0.8846722438187493}}`
- 모순: scale·correlation·spectrum·class alignment 효과를 섞지 않고 조건 간 차이를 확인한다.
- 수정된 원리: 입력 구조가 W1 gradient 정렬과 Jacobian 전달 효율을 통해 geometry gain을 결정한다.

## 2

- 가설: 상관 구조를 제거하면 W1 정렬과 geometry gain이 약해진다.
- 통제 실험: scale_only vs decorrelated PCA rotation
- 실제 결과: `{"scale_only": {"parameter_update_norm": 0.11570514471527175, "jacobian_amplification": 8.008795618446127, "same_norm_geometry_gain": 4.090195066407682, "hidden_class_distance_gap_change": 0.33271800264641127, "geometry_gain_per_hidden_norm": 0.40618673915580833, "input_gradient_alignment": 0.8933221260890971, "input_update_alignment": 0.8846722438187493}, "decorrelated": {"parameter_update_norm": 0.0927464553783629, "jacobian_amplification": 1.7189963812507765, "same_norm_geometry_gain": 0.3699062636080908, "hidden_class_distance_gap_change": 0.013333877580832123, "geometry_gain_per_hidden_norm": 0.10717461321903077, "input_gradient_alignment": 0.9536980868642368, "input_update_alignment": 0.26358610333472143}}`
- 모순: scale·correlation·spectrum·class alignment 효과를 섞지 않고 조건 간 차이를 확인한다.
- 수정된 원리: 입력 구조가 W1 gradient 정렬과 Jacobian 전달 효율을 통해 geometry gain을 결정한다.

## 3

- 가설: 고유값 spectrum 평탄화가 추가로 W1 증폭을 약화한다.
- 통제 실험: decorrelated vs eigen_flattened
- 실제 결과: `{"decorrelated": {"parameter_update_norm": 0.0927464553783629, "jacobian_amplification": 1.7189963812507765, "same_norm_geometry_gain": 0.3699062636080908, "hidden_class_distance_gap_change": 0.013333877580832123, "geometry_gain_per_hidden_norm": 0.10717461321903077, "input_gradient_alignment": 0.9536980868642368, "input_update_alignment": 0.26358610333472143}, "eigen_flattened": {"parameter_update_norm": 0.10103250167176737, "jacobian_amplification": 2.003380989876648, "same_norm_geometry_gain": 0.15277660616577354, "hidden_class_distance_gap_change": 0.0026162299804005813, "geometry_gain_per_hidden_norm": 0.013899400536419781, "input_gradient_alignment": 0.36075073703985777, "input_update_alignment": 0.3100896478644367}}`
- 모순: scale·correlation·spectrum·class alignment 효과를 섞지 않고 조건 간 차이를 확인한다.
- 수정된 원리: 입력 구조가 W1 gradient 정렬과 Jacobian 전달 효율을 통해 geometry gain을 결정한다.

## 4

- 가설: W1 gradient/update가 input class-separating subspace와 정렬될수록 geometry gain이 커진다.
- 통제 실험: 각 조건의 input class-subspace projection alignment와 geometry 지표의 비교
- 실제 결과: `{"unscaled": {"alignment": 0.675409076118939, "geometry_efficiency": 0.2724948175688706}, "scale_only": {"alignment": 0.8846722438187493, "geometry_efficiency": 0.40618673915580833}, "decorrelated": {"alignment": 0.26358610333472143, "geometry_efficiency": 0.10717461321903077}, "eigen_flattened": {"alignment": 0.3100896478644367, "geometry_efficiency": 0.013899400536419781}}`
- 모순: scale·correlation·spectrum·class alignment 효과를 섞지 않고 조건 간 차이를 확인한다.
- 수정된 원리: 입력 구조가 W1 gradient 정렬과 Jacobian 전달 효율을 통해 geometry gain을 결정한다.

## 최소 원리

`입력 covariance/eigenvalue 구조 → W1 class-direction alignment → downstream Jacobian 전달 → hidden geometry`
