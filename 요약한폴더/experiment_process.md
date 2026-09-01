# 실험 과정: 뉴런 탐색에서 iterative feedback까지

## 문제를 어떻게 쪼갰는가

처음부터 “AI가 어떻게 판단하는가?”를 직접 설명하려 하지 않았다. 질문을 다음 단위로 나눴다.

```text
개별 뉴런
→ 뉴런 pair
→ 분산된 activation pattern
→ hidden representation geometry
→ geometry를 만드는 parameter update
→ gradient update의 재계산 과정
```

각 단계에서 가설과 실제 결과가 맞지 않으면 그 모순을 다음 단계의 질문으로 사용했다.

## 1. 출발점

처음 질문은 “어떤 뉴런이나 계산이 AI 판단을 담당하는가?”였다. 가중치와 뉴런을 제거하며 중요도를 측정했지만, 개별 중요도만으로는 모델의 판단을 설명할 수 없었다.

여기서 중요한 점은 ablation 결과가 “뉴런이 전혀 중요하지 않다”는 뜻이 아니라는 것이다. 같은 기능이 여러 뉴런과 여러 parameter에 분산되어 있어, 하나를 제거하는 것만으로 기능을 설명하기 어렵다는 뜻이다.

## 2. 가설의 수정

| 단계 | 처음 가설 | 관찰 | 모순 | 수정된 질문 |
|---|---|---|---|---|
| 개별 뉴런 | 중요한 뉴런이 존재한다 | seed마다 중요한 뉴런 번호가 달랐다 | 뉴런 번호가 재현되지 않음 | 공통 representation 구조가 있는가? |
| 뉴런 pair | 고정 pair가 상호작용을 담당한다 | 일부 pair만 의미가 있었다 | seed 간 안정성이 낮았다 | 분산된 activation pattern인가? |
| activation direction | 활동별 고유 방향이 존재한다 | 방향과 class geometry가 반복됐다 | 방향의 사람이 읽을 의미는 불명확했다 | geometry를 직접 설계할 수 있는가? |
| geometry inverse | 목표 geometry를 Jacobian으로 만들 수 있다 | validation 목표는 개선됐다 | test loss와 covariance가 불안정했다 | gradient training과 무엇이 다른가? |
| gradient averaging | 평균이 공통 성분을 남긴다 | residual은 상쇄됐지만 batch 효과가 섞였다 | averaging만으로는 충분하지 않았다 | gradient 재계산 feedback이 핵심인가? |

## 3. geometry inverse의 한계

validation의 direct-overlap objective를 Jacobian inverse로 최소-norm 설계했다. 특정 validation overlap은 감소했지만, 전체 분류 loss와 test distribution geometry를 함께 보존하지 못했다.

이 결과는 사람이 원하는 geometry를 지정하는 것만으로는 부족하다는 단서였다. 전체 sample·class의 오차가 update마다 어떻게 달라지는지가 더 중요할 수 있었다.

inverse는 validation에서 정의한 국소 목표를 만족하도록 Jacobian의 최소-norm parameter update를 계산한다. 이 방식은 “원하는 hidden 이동이 가능한가?”를 확인하는 데는 유용하지만, 그 목표 자체가 test 분포의 모든 sample과 class를 대표한다고 보장하지 않는다.

## 4. gradient averaging 통제

sample별 gradient를 batch 평균 방향과 residual로 분해했다.

| Batch | 평균 방향 정렬 | residual 상쇄율 | Test accuracy |
|---:|---:|---:|---:|
| 1 | 1.000 | 0.000 | 0.359 |
| 4 | 0.488 | 0.445 | 0.423 |
| 16 | 0.309 | 0.648 | 0.548 |
| 64 | 0.265 | 0.704 | 0.640 |
| 128 | 0.249 | 0.718 | 0.670 |

평균 gradient는 residual-only와 random update보다 유용했지만, 큰 batch가 항상 좋은 것은 아니었다. 총 sample exposure와 update 횟수가 함께 변했기 때문이다.

여기서 residual 상쇄율은 `1 - 평균 gradient norm / sample gradient norm의 평균`으로 정의했다. residual 벡터의 평균은 정의상 0이므로, 이 지표는 평균을 냈을 때 원래 gradient norm 중 얼마가 사라지는지를 나타낸다.

## 5. exposure 통제

모든 batch가 동일한 1,280개 sample을 보게 했다.

결과는 batch 차이가 여전히 남는다는 것이었다. 따라서 이전 효과를 단순한 sample exposure만으로 설명할 수 없었다. 하지만 큰 batch에서는 residual 상쇄가 커지는 동시에 update 횟수와 parameter movement가 작아졌다.

## 6. 동일 sample 집합 통제

같은 128개 sample을 같은 순서로 처리하되 averaging 시점만 바꿨다. 큰 batch가 더 많은 residual을 제거했지만 성능은 오히려 낮았다. 동일한 총 parameter movement norm으로 맞춰도 큰 batch가 우세하지 않았다.

따라서 평균화는 유용한 성분을 남기지만, 그것만으로 일반화가 보장되지는 않는다는 결론을 얻었다.

이 통제는 batch size가 커질수록 더 많은 데이터를 보았다는 단순한 설명을 제거하기 위한 것이다. 다만 같은 sample 수를 보더라도 batch size가 바뀌면 Adam update 횟수와 parameter 이동량이 달라진다. 그래서 다음 단계에서 이동량을 동일하게 맞춘 counterfactual도 확인했다.

## 7. 최종 결정 실험

### 세 조건의 차이

세 방법은 모두 같은 초기 parameter에서 시작하지만, gradient 정보의 유효 기간이 다르다.

| 조건 | gradient가 계산되는 시점 | 이후 사용 방식 |
|---|---|---|
| One-shot | 초기 상태에서 128개 sample 전체 | 평균을 한 번 적용 |
| Frozen | 초기 상태에서 128개 sample 전체 | 저장된 gradient를 계속 적용 |
| Recomputed | 매 sample update 직전 | 현재 parameter에서 다시 계산 |

Frozen은 parameter가 이미 달라졌는데도 옛날 representation의 gradient를 계속 사용한다. Recomputed는 매번 현재 오차를 다시 측정한다.

같은 초기화와 같은 128개 sample에서 gradient 재계산 여부만 바꿨다.

- **One-shot**: 초기 128개 gradient의 평균을 한 번 적용
- **Frozen sequential**: 초기 gradient를 저장한 뒤 순서대로 적용
- **Recomputed sequential**: 매 update 후 현재 parameter에서 다음 gradient를 재계산

| 방법 | Parameter 이동 | Train loss | Validation loss | Test loss | Test accuracy |
|---|---:|---:|---:|---:|---:|
| One-shot | 0.196 | 1.806 | 1.802 | 1.846 | 0.279 |
| Frozen | 5.940 | 9.972 | 9.976 | 9.544 | 0.338 |
| Recomputed | 2.016 | 0.723 | 0.736 | 0.768 | 0.706 |

동일한 최종 parameter movement norm으로 맞춘 뒤에도 test accuracy는 각각 `0.356`, `0.342`, `0.698`이었다.

Recomputed 조건에서 현재 sample gradient와 같은 sample의 초기 gradient 사이 cosine은 평균 `0.350`, 마지막에는 `0.234`였다. 즉 representation이 변하면서 필요한 gradient 방향도 실제로 변했다.

연속 sample의 gradient cosine만 비교하면 sample 간 차이와 representation feedback을 구분할 수 없다. 그래서 같은 sample의 현재 gradient와 초기 gradient를 직접 비교하는 지표를 추가했다. 이 값이 낮아진 것이 iterative feedback의 직접적인 증거다.

## 8. 최종 원리

```text
현재 representation
→ 현재 오차
→ gradient
→ parameter 변화
→ representation 변화
→ 새로운 오차 구조
→ 새로운 gradient
→ 반복 trajectory
→ distribution-level geometry
→ unseen generalization
```

최종 결론은 다음과 같다.

> Gradient training이 일반화되는 핵심은 초기 gradient를 잘 계산하는 것이 아니라, representation이 바뀔 때마다 현재 상태에 맞는 gradient를 다시 계산하는 iterative feedback이다.

고정된 geometry inverse나 frozen gradient는 이전 상태의 목표를 계속 적용한다. 반면 recomputed training은 각 update가 만든 representation 변화를 다음 update의 정보로 사용한다.

## 결과를 해석할 때의 주의점

1. class separation이 커졌다고 반드시 accuracy가 좋아지는 것은 아니다. Frozen 조건처럼 표현 공간만 크게 벌어지고 decision logit이 망가질 수 있다.
2. validation/test geometry cosine이 높다고 반드시 분류 성능이 좋은 것은 아니다. 구조의 방향이 비슷해도 boundary 위치와 margin이 나쁠 수 있다.
3. 128개 sample만 사용한 최종 실험은 feedback 메커니즘을 비교하기 위한 통제 실험이다. 전체 UCI HAR 학습 성능을 보고하는 실험이 아니다.
4. 최종 결론은 “recomputed gradient가 모든 상황에서 최적”이라는 뜻이 아니라, 현재 representation을 반영하는 재계산이 고정 gradient보다 일반화에 필요한 핵심 과정이라는 뜻이다.
