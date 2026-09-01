# 최종 결과 요약

## 질문

왜 AI는 최종 가중치를 처음부터 한 번에 계산하지 않고 반복적으로 학습해야 하는가?

여기서 “한 번에 계산”은 최적해를 정확히 구하는 알고리즘과 비교한다는 뜻이 아니다. 현재 초기 representation에서 계산한 gradient 방향을 이후 상태에서도 그대로 사용할 수 있는지를 묻는 통제 실험이다.

## 비교

동일한 random initialization, 동일한 128개 train sample, 동일한 sample 순서에서 세 가지 방법을 비교했다.

```text
A. One-shot averaging
   θ0에서 128개 gradient를 계산하고 평균 gradient를 한 번 적용

B. Frozen-gradient sequential
   θ0에서 계산한 gradient를 저장하고 parameter가 바뀌어도 계속 적용

C. Recomputed sequential
   매 update 후 현재 parameter에서 다음 gradient를 재계산
```

## 핵심 수치

| 방법 | Test accuracy | Test loss | 동일 norm Test accuracy |
|---|---:|---:|---:|
| One-shot | 0.279 | 1.846 | 0.356 |
| Frozen | 0.338 | 9.544 | 0.342 |
| Recomputed | 0.706 | 0.768 | 0.698 |

## 판정

고정 방향만으로 학습이 대부분 설명된다는 H1은 기각된다. Recomputed가 frozen과 one-shot보다 일관되게 우수하므로 iterative-feedback 가설을 지지한다.

Frozen의 hidden separation은 오히려 크게 증가했지만 loss가 폭증했다. 이는 “class 중심을 멀리 만드는 geometry”와 “decision boundary에 맞는 geometry”가 다르다는 것을 보여준다. 최종 판단에서는 separation 하나가 아니라 loss, accuracy, overlap, hidden movement를 함께 사용했다.

다만 “평균 gradient가 항상 일반화한다”는 더 약한 가설도 충분하지 않다. 평균화는 충돌하는 sample-specific 성분을 줄이지만, 최종 geometry는 현재 representation에서 다시 계산된 gradient의 연속적인 trajectory가 만든다.

## 사람이 이해할 수 있는 최소 모델

```text
representation 상태
→ 분류 오차
→ 상태 의존적 gradient
→ 작은 parameter update
→ 새로운 representation 상태
→ 다시 오차 측정
```

이 반복이 class distribution을 점진적으로 분리하고, validation에만 맞춘 국소 geometry가 아니라 unseen test에도 공유되는 geometry를 만든다.

## 한 문장 결론

> 학습은 처음부터 정답 geometry를 계산하는 과정이 아니라, 현재 representation에서 생긴 오차를 이용해 다음 representation을 만들고 그 결과에 맞춰 다시 gradient를 수정하는 과정이다.
