# Reality Representation Experiment

## 목적

복잡한 센서값을 사람이 먼저 선별하지 않고 모델에 모두 입력한다. 모델이 만든 내부 표현을 역분석한 뒤, 다시 센서 현실의 언어로 되돌린다.

최종적으로 다음 사슬을 하나의 실험에서 증명한다.

```text
raw sensor values
  -> learned representation
  -> task-critical directions and relations
  -> recurring latent structure across models
  -> original sensor combinations
  -> counterfactual intervention on those combinations
  -> selective change in activity judgment
```

핵심 결과는 단순한 압축률이나 정확도가 아니다. 다음 문장을 데이터로 방어할 수 있어야 한다.

> 모델은 센서값 전체에서 특정한 관계 구조를 만들고, 그 구조가 실제 활동 판별에 필요하며, 원래 센서 조합에 개입하면 판단이 선택적으로 변한다.

## 범위와 데이터

- 데이터: 기존 `uci_har/UCI HAR Dataset/`의 raw inertial signals.
- 입력: 9개 채널 × 128 timestep 전체값.
  - body acceleration: x/y/z
  - total acceleration: x/y/z
  - body gyroscope: x/y/z
- 모델 입력에는 사람이 정한 feature subset을 사용하지 않는다.
- 기존 561개 engineered feature는 본 실험의 입력이 아니라 비교·해석용 보조 자료로만 사용한다.
- 분할: subject 기준 train/validation/test. test subject와 label은 노드·축·dictionary 선택에 사용하지 않는다.
- 반복: 최소 5개 seed. 모델 선택, sparse 해석, intervention threshold는 train/validation에서만 결정한다.

## 단계별 계획

### 1. 전체 raw 입력으로 기준 모델 학습

파일: `raw_model.py`

- 9 × 128 raw window를 받는 작은 temporal CNN 또는 동일한 1D convolution encoder를 구현한다.
- 마지막 classifier 직전의 bottleneck을 `h ∈ R^32`로 고정한다.
- 모델은 모든 raw channel을 동시에 받는다.
- 저장물:
  - seed별 checkpoint
  - train/validation/test accuracy
  - sample별 `h`, logits, prediction
  - 학습 설정과 subject split manifest

통과 기준:

- test accuracy를 보고하되, 정확도 자체를 주된 주장으로 사용하지 않는다.
- 모든 이후 선택은 validation에서만 수행된다.

### 2. 분류에 필요한 내부 방향과 최소 차원 측정

파일: `latent_subspace.py`

- `h`의 선형 방향을 찾아 `z = W h`로 투영한다.
- `k = 1, 2, 4, 8, 16, 32`를 비교한다.
- 비교 대상:
  1. learned task-preserving subspace
  2. PCA subspace
  3. random orthogonal subspace
  4. label-only supervised projection
- 각 차원에서 accuracy, macro-F1, class-pair confusion, logit margin을 기록한다.
- 최소 차원은 "원래 test accuracy의 95% 이상을 validation에서 유지하는 가장 작은 k"로 사전 정의한다.

결과 질문:

- 분류에 실제로 필요한 방향 수는 몇 개인가?
- 그 방향은 단순 variance 방향인가, task-specific 방향인가?
- 어떤 activity pair가 먼저 무너지는가?

### 3. 값이 아니라 관계가 필요한지 검증

파일: `relation_breaking.py`

동일한 raw sample과 동일한 marginal distribution을 유지하면서 내부 또는 입력의 관계만 파괴한다.

- temporal order shuffle: 각 채널의 값 분포는 유지하고 시간 관계 파괴
- cross-channel pairing shuffle: 채널별 주변 분포는 유지하고 채널 간 동시 관계 파괴
- phase-preserving control: Fourier phase 또는 local temporal structure를 보존하는 대조군
- orthogonal latent rotation: 좌표값은 재표현하되 subspace는 보존
- relation-preserving projection: 같은 `k`에서 관계를 최대한 보존하는 투영

각 조작에 대해:

- accuracy 변화
- activity별 변화
- confusion transition
- latent distance와 class margin 변화

통과 기준:

- "관계가 중요하다"는 주장은 marginal-preserving relation destruction이 matched control보다 큰 성능 손상을 만들 때만 채택한다.
- 어떤 조작도 손상을 만들지 않으면 관계 중심 가설을 기각하고, 값 기반 또는 다른 구조를 보고한다.

### 4. latent 방향을 원래 센서 조합으로 번역

파일: `sensor_dictionary.py`

모델 입력을 미리 선별하지 않는다. dictionary는 학습 입력이 아니라 사후 해석 언어다.

전체 raw signal에서 다음 후보 표현을 계산한다.

- channel별 level, slope, variance, energy
- 짧은/긴 window의 변화량
- axis 간 correlation 및 covariance
- acceleration–gyro coupling
- magnitude와 axis ratio
- peak, zero-crossing, dominant frequency, spectral band energy
- 시간 지연 correlation과 phase difference

각 latent direction `z_j`에 대해 train/validation에서 sparse regression을 적합한다.

```text
z_j ≈ Σ_l a[j,l] * dictionary_l(raw_signal)
```

- L1 regularization으로 설명 조합을 희소화한다.
- 동일한 설명력을 가진 조합은 더 짧은 조합을 선택한다.
- 설명도(`R²`, held-out reconstruction error, coefficient stability)를 함께 기록한다.
- 결과는 "센서 A 값이 크다"가 아니라 "센서 A와 B의 시간적 관계가 변한다" 형태로 출력한다.

### 5. recovered combination에 직접 counterfactual intervention

파일: `sensor_intervention.py`

사후 설명이 실제 원인 후보인지 검증한다.

각 recovered sensor combination에 대해 raw window의 해당 구조만 바꾼다.

- cross-channel relation intervention: 한 채널의 phase/time alignment를 바꾼다.
- temporal relation intervention: slope 또는 periodicity만 바꾼다.
- magnitude-preserving intervention: 전체 energy는 유지하고 axis/coupling relation만 바꾼다.
- matched random control: 같은 크기의 무작위 조작을 다른 dictionary 항목에 적용한다.

판정량:

- target activity pair의 logit margin 변화
- 전체 accuracy 변화
- 비표적 activity pair의 변화
- intervention dose-response
- 원래 sample의 prediction이 target 방향으로 움직인 비율

통과 기준:

- recovered combination intervention이 matched random control보다 target pair에 선택적인 효과를 보여야 한다.
- 효과가 없으면 해당 조합은 설명이 아니라 상관 신호로 표시한다.
- raw signal의 물리적 범위를 벗어난 조작은 버린다.

### 6. 서로 다른 모델에서 반복되는 구조 확인

파일: `cross_model_recurrence.py`

seed만 바꾸지 말고 최소 두 encoder 형태를 사용한다.

- temporal CNN
- compact temporal MLP 또는 다른 kernel width CNN

모델 간 비교:

- CKA 또는 centered kernel alignment
- Procrustes/CCA 정렬 후 latent direction similarity
- sparse sensor dictionary overlap
- intervention effect 방향과 크기
- activity pair별 recurrence rate

반복 구조의 정의:

- 두 모델군 이상에서 dictionary 상위 조합이 재등장
- coefficient sign/relative weight가 안정적
- held-out intervention이 같은 activity pair를 같은 방향으로 바꿈

단일 seed에서만 나타난 축은 발견이 아니라 모델 특이 결과로 분류한다.

### 7. 보고서 생성

파일: `build_report.py`, 결과: `REPORT.md`

최종 표의 한 행은 다음 정보를 모두 포함한다.

| activity pair | minimum k | latent relation | original sensor combination | intervention effect | model recurrence |
|---|---:|---|---|---:|---:|

보고서 순서:

1. 현실: raw sensor window가 무엇인지
2. 학습: 모델이 무엇을 보도록 강제하지 않았는지
3. 표현: 어떤 latent directions와 관계가 생겼는지
4. 붕괴: 어떤 관계를 깨면 어떤 판정이 무너지는지
5. 역변환: 그 구조를 만든 원래 센서 조합
6. 개입: 조합을 바꿨을 때 판단이 선택적으로 변하는지
7. 반복성: 다른 모델에서도 재현되는지
8. 한계: dictionary completeness, intervention realism, subject generalization

## 디렉터리 계약

```text
uci_har/reality_representation/
├── PLAN.md
├── raw_model.py
├── latent_subspace.py
├── relation_breaking.py
├── sensor_dictionary.py
├── sensor_intervention.py
├── cross_model_recurrence.py
├── build_report.py
├── test_reality_representation.py
├── manifests/
├── checkpoints/
├── results/
└── REPORT.md
```

- 모든 실행은 이 디렉터리에서 상대경로로 재현 가능해야 한다.
- 원본 `uci_har` 실험 파일은 수정하지 않는다.
- 중간 결과는 JSON/NPZ로 저장하고, 보고서는 결과 파일에서만 생성한다.
- test subject를 이용한 선택이나 설명 조합 fitting은 금지한다.

## 성공·실패 기준

성공은 정확도가 높다는 뜻이 아니다. 다음 네 조건을 모두 만족해야 한다.

1. 전체 raw input으로 학습한 모델에서 validation 기준 최소 latent dimension을 찾는다.
2. relation destruction이 marginal-preserving control보다 분류를 선택적으로 훼손한다.
3. latent direction이 held-out data에서 안정적인 sensor combination으로 번역된다.
4. 번역된 combination에 개입했을 때 matched random control보다 예측이 특정 activity pair 방향으로 움직이고, 다른 모델에서도 반복된다.

하나라도 실패하면 결론을 낮춘다.

- 1 실패: 최소 표현 차원 주장을 하지 않는다.
- 2 실패: 관계보다 값 또는 다른 구조가 중요하다고 보고한다.
- 3 실패: latent 구조는 발견했지만 현실 언어로 해석하지 못했다고 보고한다.
- 4 실패: 설명은 상관관계이며 원인 후보로 승격하지 않는다.

## 기존 결과와의 경계

기존 561-feature bottleneck·node ablation 결과는 이 새 폴더의 본 실험 결과로 재사용하지 않는다. 그것들은 "학습된 표현을 줄였을 때 무엇이 남는가"를 보여주는 선행 탐색이다. 새 실험은 raw sensor 전체 입력에서 시작해, 표현을 만들고, 관계를 깨고, 원래 센서 조합에 개입하는 폐루프를 새로 검증한다.
