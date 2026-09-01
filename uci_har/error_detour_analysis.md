# 오차 최소화만 vs 오차 증가 후 감소

두 조건은 같은 초기 가중치, 데이터 분할, seed, 총 20회 Adam 업데이트를 사용했다. 기준군은 20회 모두 cross-entropy를 줄이고, 우회군은 10회 gradient ascent로 오차를 키운 뒤 10회 gradient descent로 오차를 줄였다.

## minimize_only
- train loss: 2.379 → 0.745 → 0.448
- validation loss: 2.373 → 0.743 → 0.447
- test loss: 2.390 → 0.801 → 0.502
- test accuracy: 0.167 → 0.670 → 0.812

## ascent_then_descent
- train loss: 2.379 → 16.813 → 13.596
- validation loss: 2.373 → 16.862 → 13.656
- test loss: 2.390 → 16.151 → 13.008
- test accuracy: 0.167 → 0.044 → 0.029

## 해석 기준

오차를 줄이는 방향은 현재 데이터의 정답 확률을 높이는 방향이다. 오차를 키우는 단계는 같은 목적함수의 반대 방향이므로, 특별한 탐색 효과가 없다면 제한된 업데이트 예산에서 기준군보다 불리하거나 회복 시간이 필요할 것으로 예상한다. 최종 비교는 test를 학습에 사용하지 않고 마지막 update의 test loss/accuracy로 판단한다.
