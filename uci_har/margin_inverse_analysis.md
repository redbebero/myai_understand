# Decision-boundary aligned hidden geometry inverse design

기존 UCI HAR 561→64→32→6 MLP를 seed별 80 epoch 학습하고, 각 class의 최저-margin sample과 class centroid를 정답 class의 output-weight 방향 안쪽으로 이동시키는 목표를 정의했다.

## 결과

- baseline: accuracy=0.943, mean margin=14.520, q10=3.440, min=-27.266
- margin_inverse: accuracy=0.941, mean margin=14.745, q10=3.399, min=-27.558, separation ratio=3.793, gate change=0.5%
- centroid_inverse: accuracy=0.935, mean margin=14.872, q10=3.179, min=-24.825, separation ratio=3.509, gate change=5.5%
- random_same_norm: accuracy=0.943, mean margin=14.520, q10=3.458, min=-27.288, separation ratio=3.778, gate change=0.1%
- gradient_same_norm: accuracy=0.939, mean margin=14.447, q10=3.270, min=-25.408, separation ratio=3.788, gate change=0.9%

## 계산 흐름

`decision boundary → margin deficit → Δh_target ∥ (w_y−w_j) → Jθ pseudoinverse → Δθ → actual margin/accuracy`

margin inverse가 centroid inverse보다 분류 경계와 직접 정렬된 목표를 사용했는지, margin·accuracy·confusion의 seed별 원자료를 함께 비교해야 한다.

## 반복 inverse

- accuracy: 0.942, 0.942, 0.942, 0.942, 0.942
- mean margin: 14.567, 14.618, 14.667, 14.722, 14.773
- q10 margin: 3.452, 3.421, 3.426, 3.429, 3.438

## 해석

목표 margin이 실제로 증가하지만 accuracy가 증가하지 않으면, sample-level margin 목표와 전체 test distribution 또는 output-layer 정렬 사이에 불일치가 있다는 뜻이다. 반복 inverse의 개선·악화와 gate 변화는 local Jacobian 역산의 적용 범위를 보여준다.
