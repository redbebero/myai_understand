# Jacobian inverse design으로 hidden geometry 만들기

기존 561→64→32→6 MLP를 seed별 80 epoch 학습한 뒤, hidden2의 세 동적 활동 centroid를 중심에서 1.2배 확대하는 목표를 정의했다. 새 feature/neuron은 사용하지 않았다.

## 전체 결과

- baseline: test accuracy 0.943, target distance error 4.024
- inverse one-shot: target distance error 1.012, mean pair distance 23.074, accuracy 0.935
- random same-norm: target distance error 4.079, accuracy 0.941
- gradient same-norm: target distance error 5.055, accuracy 0.577

inverse는 baseline 대비 목표 geometry 오차를 약 85% 줄였고, random은 거의 줄이지 못했다. 그러나 accuracy는 0.943에서 0.935로 소폭 감소했다.

## 1. 목표 geometry

- 가설: centroid pair distance를 20% 확대하면 목표가 class separation을 명확히 지정한다.
- 비교 기준: 현재 test centroid geometry와 중심 기준 1.2배 확대한 target geometry.
- 모순/실패 원인: ReLU gate 변화, Jacobian 선형근사, parameter 자유도, test generalization을 원자료에서 분리한다.
- 수정된 원리: centroid-level geometry는 local Jacobian 범위 안에서만 역설계 가능하다.

## 2. 필요한 Δh

- 가설: 현재 centroid와 목표 centroid 차이가 필요한 hidden 이동을 정의한다.
- 비교 기준: centroid 차이의 평균 제곱근 크기와 pair-distance 구조.
- 모순/실패 원인: ReLU gate 변화, Jacobian 선형근사, parameter 자유도, test generalization을 원자료에서 분리한다.
- 수정된 원리: centroid-level geometry는 local Jacobian 범위 안에서만 역설계 가능하다.

## 3. 역계산 Δθ

- 가설: centroid Jacobian의 regularized pseudoinverse가 최소-norm parameter update를 제공한다.
- 실제 결과: J shape=[96, 38017], 유효 rank 평균=85.8/96, 선형 target residual=0.0044 (target Δh 평균 크기=0.5063).
- 모순/실패 원인: ReLU gate 변화, Jacobian 선형근사, parameter 자유도, test generalization을 원자료에서 분리한다.
- 수정된 원리: centroid-level geometry는 local Jacobian 범위 안에서만 역설계 가능하다.

## 4. 실제 적용

- 가설: inverse update가 random/gradient 동일 norm보다 목표 geometry에 가까워진다.
- 실제 결과: inverse가 동일 norm random/gradient보다 target geometry에 가까워졌다. 다만 고정된 output layer를 다시 맞추지 않았으므로 geometry 조작이 accuracy 상승으로 이어지지는 않았다.
- 모순/실패 원인: ReLU gate 변화, Jacobian 선형근사, parameter 자유도, test generalization을 원자료에서 분리한다.
- 수정된 원리: centroid-level geometry는 local Jacobian 범위 안에서만 역설계 가능하다.

## 5. 반복 inverse

- 가설: Jacobian을 갱신하면 작은 inverse step이 목표에 안정적으로 접근한다.
- 실제 결과: 반복 inverse distance error=3.313, 3.209, 2.754, 2.759, 2.628; accuracy=0.942, 0.923, 0.925, 0.918, 0.911.
- 모순/실패 원인: ReLU gate 변화, Jacobian 선형근사, parameter 자유도, test generalization을 원자료에서 분리한다.
- 수정된 원리: centroid-level geometry는 local Jacobian 범위 안에서만 역설계 가능하다.

## 실패 원인 분해

- Jacobian 근사: 선형 residual은 작아 local tangent space 자체는 target Δh를 거의 표현했다.
- ReLU gate: one-shot inverse에서 평균 6.4%의 gate가 변해, 큰 update에서는 고정-gate Jacobian이 완전한 예측이 아니다.
- 목표 geometry의 의미: centroid 거리만 확대했기 때문에 기존 output weight와의 정렬·sample-level 분류 경계는 직접 최적화하지 않았다. 따라서 geometry 성공과 accuracy 성공은 분리됐다.
- parameter 자유도: 유효 rank가 대부분 82~96/96이므로 이번 실패를 자유도 부족으로 설명하기 어렵다. seed 31처럼 inverse norm과 gate 변화가 큰 경우에는 local 선형성 붕괴가 더 유력하다.
- 반복 inverse: 작은 step은 대체로 목표에 접근했지만 seed 31에서 발산성 흔들림이 나타나, Jacobian 재계산만으로 전역 목표를 보장하지 않는다.

## 수정된 최소 메커니즘

`목표 centroid geometry → Δh_target → local Jθ pseudoinverse → hidden parameter update → ReLU-gated actual movement → geometry 변화`

결론: Jacobian 역산은 ‘원하는 hidden centroid 구조를 국소적으로 만드는 도구’로는 작동한다. 하지만 그 구조가 곧 decision boundary나 정확도를 의미하지는 않는다. 정확도까지 역설계하려면 목표에 output-weight 정렬 또는 class loss 조건을 함께 포함해야 한다.
