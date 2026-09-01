# Jacobian으로 설명하는 weight update와 hidden 이동

## 1

- 수식의 예측: Δh ≈ Jθ(h)Δθ로 실제 hidden 이동을 설명할 수 있다.
- 실제 측정: Adam 실제 update의 Jacobian 예측 Δh와 실제 Δh의 cosine, R², norm error를 계산한다.
- 실제 결과: {'adam_cosine': 0.9707903329178502, 'adam_r2': 0.9146906934643547, 'adam_norm_error': 0.2373880028992693, 'adam_predicted_gap': 0.3533967255425479, 'adam_actual_gap': 0.3654070570907459}
- 맞지 않는 점: batch 원자료에서 선형근사 오차와 gate 변화 효과를 확인한다.
- 수정된 메커니즘: `loss → Δθ → Jθ(h)Δθ → 실제 Δh → class 거리 변화`의 각 오차원을 분리한다.

## 2

- 수식의 예측: W1/W2/bias contribution을 합치면 class geometry 변화도 예측된다.
- 실제 측정: parameter별 1차 contribution과 same/different class distance 변화량을 계산한다.
- 실제 결과: {'w1': {'norm': 0.9540055108080319, 'distance_gap': 0.3323211702635261}, 'b1': {'norm': 0.0032930890597011385, 'distance_gap': 0.0002867426915021212}, 'w2': {'norm': 0.0884915810429181, 'distance_gap': 0.01696699553737113}, 'b2': {'norm': 0.002728412667829057, 'distance_gap': 5.631359634859993e-05}}
- 맞지 않는 점: batch 원자료에서 선형근사 오차와 gate 변화 효과를 확인한다.
- 수정된 메커니즘: `loss → Δθ → Jθ(h)Δθ → 실제 Δh → class 거리 변화`의 각 오차원을 분리한다.

## 3

- 수식의 예측: ReLU gate가 유지된 sample에서는 근사가 더 정확하다.
- 실제 측정: gate 유지/변경 sample의 Jacobian fit을 분리한다.
- 실제 결과: {'gate_changed_fraction': 0.9144186046511628, 'stable_gate_r2': 0.8420603549761105, 'changed_gate_r2': 0.9116994265197741}
- 맞지 않는 점: batch 원자료에서 선형근사 오차와 gate 변화 효과를 확인한다.
- 수정된 메커니즘: `loss → Δθ → Jθ(h)Δθ → 실제 Δh → class 거리 변화`의 각 오차원을 분리한다.

## 4

- 수식의 예측: Adam의 preconditioning이 단순 SGD와 representation 이동을 다르게 만든다.
- 실제 측정: 동일 gradient에서 Adam 실제 update와 SGD counterfactual의 fit·거리 변화를 비교한다.
- 실제 결과: {'sgd_cosine': 0.9988891508128256, 'sgd_r2': 0.9968719701112932, 'sgd_norm_error': 0.042618317142949856, 'adam_w1_gap': 0.3323211702635261, 'adam_w2_gap': 0.01696699553737113}
- 맞지 않는 점: batch 원자료에서 선형근사 오차와 gate 변화 효과를 확인한다.
- 수정된 메커니즘: `loss → Δθ → Jθ(h)Δθ → 실제 Δh → class 거리 변화`의 각 오차원을 분리한다.

## 최소 메커니즘

`loss gradient → Adam/SGD parameter update → local Jacobian prediction → ReLU-gated nonlinear residual → class geometry`
