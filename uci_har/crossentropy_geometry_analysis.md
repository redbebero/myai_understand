# Cross-entropy gradient와 hidden geometry

## 1

- 수식의 예측: -∂L/∂h = W^T(y-p)는 정답 class weight 방향 성분을 가진다.
- 실제 측정: 정답 weight와 negative gradient/actual Δh의 cosine을 batch마다 측정한다.
- 실제 결과: {'target_weight_vs_negative_gradient': 0.7690661313431374, 'target_weight_vs_actual_delta': 0.2165659206029133}
- 맞지 않는 점: batch별 원자료에서 cosine과 정렬이 완전하지 않은 부분을 확인한다.
- 수정된 메커니즘: `cross-entropy error → output weights → hidden movement → geometry → boundary`의 어느 단계가 약한지 구분한다.

## 2

- 수식의 예측: 오답 class 확률이 높을수록 해당 weight 방향의 음의 성분이 강해진다.
- 실제 측정: wrong probability와 wrong-weight cosine의 상관 및 상·하위 quartile을 비교한다.
- 실제 결과: {'wrong_weight_vs_negative_gradient': -0.5637527725830854, 'wrong_probability_gradient_correlation': -0.3573319934321142, 'high_vs_low_wrong_probability_cosine': [-0.6423645286568981, -0.506209566484765]}
- 맞지 않는 점: batch별 원자료에서 cosine과 정렬이 완전하지 않은 부분을 확인한다.
- 수정된 메커니즘: `cross-entropy error → output weights → hidden movement → geometry → boundary`의 어느 단계가 약한지 구분한다.

## 3

- 수식의 예측: 학습이 진행되면 class centroid가 자기 output weight와 정렬된다.
- 실제 측정: checkpoint별 centroid-weight cosine과 weight/centroid pair-angle 구조를 추적한다.
- 실제 결과: {'centroid_weight_cosine_initial': 0.08097483038961181, 'centroid_weight_cosine_final': 0.3061687554320801}
- 맞지 않는 점: batch별 원자료에서 cosine과 정렬이 완전하지 않은 부분을 확인한다.
- 수정된 메커니즘: `cross-entropy error → output weights → hidden movement → geometry → boundary`의 어느 단계가 약한지 구분한다.

## 4

- 수식의 예측: weight geometry와 centroid geometry가 함께 형성된다.
- 실제 측정: 두 pairwise cosine 행렬의 상관과 seed 간 반복성을 비교한다.
- 실제 결과: {'weight_centroid_pair_correlation_initial': -0.009464102638425925, 'weight_centroid_pair_correlation_final': 0.07482408953604798, 'mean_abs_weight_pair_cosine_change': 0.011944743811386728, 'mean_abs_centroid_pair_cosine_change': 0.26906618405804134}
- 맞지 않는 점: batch별 원자료에서 cosine과 정렬이 완전하지 않은 부분을 확인한다.
- 수정된 메커니즘: `cross-entropy error → output weights → hidden movement → geometry → boundary`의 어느 단계가 약한지 구분한다.

## 최소 메커니즘

`W^T(p-y) → Adam weight update → Δh → class centroid/weight geometry → decision boundary`
