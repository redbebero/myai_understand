# 계층적 활동 판단 분석

이 보고서는 센서 feature → 동적/정적 중간 판단 → 세부 활동 판단 가설을 단계별로 검증한다.

## Seed 7

### dynamic_static_ablation

- 가설: 중간 판단 pair는 6개 class에 공통 영향을 준다.
- 방법: test에서 pair 제거 후 class별 정확도와 confusion 비교
- 예상: 비슷한 정확도 저하
- 실제: `[0.002016129032258007, -0.04883227176220806, 0.021428571428571463, 0.004073319755600768, 0.0, 0.0]`
- 불일치: class별 저하가 다르면 공통 회로 가설이 약해진다.
- 다음 질문: pair가 특정 class 또는 동적/정적 경계에 편향되는가?

### dynamic_subset

- 가설: 동적 3개 class에는 별도의 pair가 있다.
- 방법: 동적 train subset에서 pair 선택, 동적 test subset에서 검증
- 예상: 새 pair와 feature 조건
- 실제: `{"layer": 2, "pair": [25, 1]}`
- 불일치: 기존 pair와 같으면 별도 세부 회로 가설이 약해진다.
- 다음 질문: 새 pair가 세 동적 활동 중 무엇을 구분하는가?

### feature_intervention

- 가설: 새 pair feature를 교란하면 무작위보다 출력이 더 변한다.
- 방법: 조건 feature를 0으로 교란하고 동일 크기 무작위 대조
- 예상: 조건 교란 손실 변화가 대조군보다 큼
- 실제: `{"matched_count": 562, "accuracy_before": 0.9531362653208363, "accuracy_after": 0.9545782263878875, "loss_change": -0.0018911181365534324, "random_control_loss_change": -0.0008442939337601485}`
- 불일치: 대조군보다 작거나 음수면 feature 조건의 인과 해석이 약해진다.
- 다음 질문: 더 분리된 validation split에서도 유지되는가?

### hierarchy

- 가설: 세부 pair는 동적/정적 pair의 downstream 계산이다.
- 방법: layer 순서, feature 중복, 공동 활성화 조건부 확률 비교
- 예상: downstream layer와 조건부 활성화/feature 연결
- 실제: `{"downstream_layer_order": false, "same_layer": true, "feature_overlap": [], "dynamic_pair": [2, 25, 1], "detail_pair": [2, 25, 1], "test_detail_given_dynamic": 1.0}`
- 불일치: 구조적 downstream이 아니거나 overlap이 없으면 계층 가설은 미확정이다.
- 다음 질문: 같은 layer에서 기능적 계층을 어떻게 정의할 것인가?

### hierarchical_rule

- 가설: 센서 feature→중간 판단→세부 활동 규칙으로 원래 판단을 근사할 수 있다.
- 방법: dynamic rule 뒤 detail rule을 적용한 end-to-end 평가
- 예상: 6-class와 동적 subset 모두 높은 일치
- 실제: `{"accuracy": 0.3505259586019681, "teacher_agreement": 0.3498473023413641, "dynamic_accuracy": 0.9735324058364438, "dynamic_subset_accuracy": 0.357606344628695, "fallback_static_class": 5, "fallback_dynamic_class": 0, "detail_rule_class": 0, "dynamic_rule_matches": 1309, "detail_rule_matches": 562}`
- 불일치: 6-class 정확도가 낮으면 계층 구조가 완전한 설명은 아니다.
- 다음 질문: 세부 활동을 구분하는 더 풍부한 temporal feature가 필요한가?

### 주요 pair와 feature

- 동적/정적 pair: layer 2 (25, 1)
- 세부 동적 pair: layer 2 (25, 1)
- 동적/정적 ablation class 결과: `{"0": {"baseline_accuracy": 0.9919354838709677, "ablated_accuracy": 0.9939516129032258, "accuracy_delta": 0.002016129032258007}, "1": {"baseline_accuracy": 0.9490445859872612, "ablated_accuracy": 0.9002123142250531, "accuracy_delta": -0.04883227176220806}, "2": {"baseline_accuracy": 0.9119047619047619, "ablated_accuracy": 0.9333333333333333, "accuracy_delta": 0.021428571428571463}, "3": {"baseline_accuracy": 0.890020366598778, "ablated_accuracy": 0.8940936863543788, "accuracy_delta": 0.004073319755600768}, "4": {"baseline_accuracy": 0.9605263157894737, "ablated_accuracy": 0.9605263157894737, "accuracy_delta": 0.0}, "5": {"baseline_accuracy": 0.9515828677839852, "ablated_accuracy": 0.9515828677839852, "accuracy_delta": 0.0}}`
- 세부 pair feature: tBodyAccJerkMag-entropy(), fBodyBodyGyroMag-meanFreq(), tGravityAcc-arCoeff()-X,1

## Seed 11

### dynamic_static_ablation

- 가설: 중간 판단 pair는 6개 class에 공통 영향을 준다.
- 방법: test에서 pair 제거 후 class별 정확도와 confusion 비교
- 예상: 비슷한 정확도 저하
- 실제: `[0.0, 0.006369426751592355, 0.014285714285714235, 0.006109979633401208, -0.015037593984962405, 0.0018621973929237035]`
- 불일치: 큰 모순 없음
- 다음 질문: pair가 특정 class 또는 동적/정적 경계에 편향되는가?

### dynamic_subset

- 가설: 동적 3개 class에는 별도의 pair가 있다.
- 방법: 동적 train subset에서 pair 선택, 동적 test subset에서 검증
- 예상: 새 pair와 feature 조건
- 실제: `{"layer": 2, "pair": [19, 31]}`
- 불일치: 새 pair 관찰
- 다음 질문: 새 pair가 세 동적 활동 중 무엇을 구분하는가?

### feature_intervention

- 가설: 새 pair feature를 교란하면 무작위보다 출력이 더 변한다.
- 방법: 조건 feature를 0으로 교란하고 동일 크기 무작위 대조
- 예상: 조건 교란 손실 변화가 대조군보다 큼
- 실제: `{"matched_count": 467, "accuracy_before": 0.9610670511896179, "accuracy_after": 0.9538572458543619, "loss_change": 0.020921769745310265, "random_control_loss_change": 0.006212842702514709}`
- 불일치: 조건 교란이 대조군보다 큼
- 다음 질문: 더 분리된 validation split에서도 유지되는가?

### hierarchy

- 가설: 세부 pair는 동적/정적 pair의 downstream 계산이다.
- 방법: layer 순서, feature 중복, 공동 활성화 조건부 확률 비교
- 예상: downstream layer와 조건부 활성화/feature 연결
- 실제: `{"downstream_layer_order": false, "same_layer": true, "feature_overlap": [], "dynamic_pair": [2, 9, 21], "detail_pair": [2, 19, 31], "test_detail_given_dynamic": 0.5271948608137045}`
- 불일치: 구조적 downstream이 아니거나 overlap이 없으면 계층 가설은 미확정이다.
- 다음 질문: 같은 layer에서 기능적 계층을 어떻게 정의할 것인가?

### hierarchical_rule

- 가설: 센서 feature→중간 판단→세부 활동 규칙으로 원래 판단을 근사할 수 있다.
- 방법: dynamic rule 뒤 detail rule을 적용한 end-to-end 평가
- 예상: 6-class와 동적 subset 모두 높은 일치
- 실제: `{"accuracy": 0.33864947404139806, "teacher_agreement": 0.328469630132338, "dynamic_accuracy": 0.9908381404818459, "dynamic_subset_accuracy": 0.33958183129055514, "fallback_static_class": 5, "fallback_dynamic_class": 0, "detail_rule_class": 1, "dynamic_rule_matches": 1414, "detail_rule_matches": 621}`
- 불일치: 6-class 정확도가 낮으면 계층 구조가 완전한 설명은 아니다.
- 다음 질문: 세부 활동을 구분하는 더 풍부한 temporal feature가 필요한가?

### 주요 pair와 feature

- 동적/정적 pair: layer 2 (9, 21)
- 세부 동적 pair: layer 2 (19, 31)
- 동적/정적 ablation class 결과: `{"0": {"baseline_accuracy": 0.9919354838709677, "ablated_accuracy": 0.9919354838709677, "accuracy_delta": 0.0}, "1": {"baseline_accuracy": 0.9532908704883227, "ablated_accuracy": 0.9596602972399151, "accuracy_delta": 0.006369426751592355}, "2": {"baseline_accuracy": 0.9333333333333333, "ablated_accuracy": 0.9476190476190476, "accuracy_delta": 0.014285714285714235}, "3": {"baseline_accuracy": 0.8818737270875764, "ablated_accuracy": 0.8879837067209776, "accuracy_delta": 0.006109979633401208}, "4": {"baseline_accuracy": 0.9605263157894737, "ablated_accuracy": 0.9454887218045113, "accuracy_delta": -0.015037593984962405}, "5": {"baseline_accuracy": 0.9497206703910615, "ablated_accuracy": 0.9515828677839852, "accuracy_delta": 0.0018621973929237035}}`
- 세부 pair feature: tBodyAccMag-arCoeff()1, tGravityAccMag-arCoeff()1, tBodyAccMag-arCoeff()2

## 종합 결론

현재 증거는 동적/정적 중간 판단과 동적 3개 활동 내부의 추가 계산이 존재할 가능성을 지지한다. 그러나 계층 규칙만으로 6개 활동 전체를 높은 정확도로 재현하지 못하면, 이는 완성된 해석이 아니라 부분적인 기능 분해다. 특히 pair가 다른 layer의 downstream 계산인지, 같은 layer의 병렬 계산인지 구분해야 한다.

## 재현

`python -m uci_har.hierarchical_experiment`
