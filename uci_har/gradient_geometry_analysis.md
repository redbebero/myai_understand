# Gradient descent와 class-separating geometry 형성

## 1

- 가설: loss gradient가 weight를 바꾸고 그 결과 hidden representation을 움직인다.
- 실험: 각 batch update 전후의 loss, weight update cosine, hidden2 이동량을 기록한다.
- 실제 결과: `"각 update의 loss와 gradient/update cosine은 runs.records에 저장된다."`
- 모순: batch 원자료에서 gradient 방향과 geometry 변화가 일치하지 않는 경우를 확인한다.
- 수정된 원리: loss gradient가 weight를 거쳐 representation geometry와 최종 accuracy로 전파되는 최소 경로로 정리한다.

## 2

- 가설: gradient update는 같은 class를 가깝게 하고 다른 class를 멀게 한다.
- 실험: batch 내 same/different class pair distance 변화와 distance gap 변화를 측정한다.
- 실제 결과: `"각 update의 same/different distance 변화는 runs.records.representation에 저장된다."`
- 모순: batch 원자료에서 gradient 방향과 geometry 변화가 일치하지 않는 경우를 확인한다.
- 수정된 원리: loss gradient가 weight를 거쳐 representation geometry와 최종 accuracy로 전파되는 최소 경로로 정리한다.

## 3

- 가설: representation geometry가 accuracy보다 먼저 형성된다.
- 실험: 고정 probe에서 accuracy, separation ratio, class-subspace 개입 효과의 update onset을 비교한다.
- 실제 결과: `[{"seed": 7, "accuracy": 1, "separation_ratio": 2, "subspace_effect": 1}, {"seed": 11, "accuracy": 2, "separation_ratio": 5, "subspace_effect": 2}, {"seed": 19, "accuracy": 1, "separation_ratio": 2, "subspace_effect": 1}, {"seed": 23, "accuracy": 1, "separation_ratio": 3, "subspace_effect": 1}, {"seed": 31, "accuracy": 1, "separation_ratio": 3, "subspace_effect": 1}]`
- 모순: batch 원자료에서 gradient 방향과 geometry 변화가 일치하지 않는 경우를 확인한다.
- 수정된 원리: loss gradient가 weight를 거쳐 representation geometry와 최종 accuracy로 전파되는 최소 경로로 정리한다.

## 4

- 가설: 이 변화 순서는 seed와 무관하게 반복된다.
- 실험: 5개 seed의 batch trace와 onset을 비교한다.
- 실제 결과: `"seed별 원자료와 onset을 비교한다."`
- 모순: batch 원자료에서 gradient 방향과 geometry 변화가 일치하지 않는 경우를 확인한다.
- 수정된 원리: loss gradient가 weight를 거쳐 representation geometry와 최종 accuracy로 전파되는 최소 경로로 정리한다.

## 최소 메커니즘

`loss gradient → weight update → hidden representation 이동 → class-separating geometry → accuracy`

세부 batch 기록은 JSON의 runs.records에 보존한다.
