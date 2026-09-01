# Gradient training vs geometry inverse

동일한 random initialization에서 Adam cross-entropy training은 train split mini-batch를 사용하고, inverse는 validation 전체의 기존 direct-overlap objective를 사용했다. test는 update에 사용하지 않고 trajectory 기록용 평가만 수행했다.

## Update 0 → 10
### gradient
- train loss: 2.379 → 0.740
- validation loss: 2.373 → 0.739
- test loss: 2.390 → 0.796
- validation overlap: 0.486 → 0.402
- test overlap: 0.487 → 0.411
- test boundary overlap: 0.911 → 0.663
- test boundary variance: 0.195 → 0.199
- validation/test direction cosine at end: 0.845
- class movement cosine at end: 0.135
- update-data coverage at end: 0.223

### inverse
- train loss: 2.379 → 3.106
- validation loss: 2.373 → 3.124
- test loss: 2.390 → 2.985
- validation overlap: 0.486 → 0.351
- test overlap: 0.487 → 0.390
- test boundary overlap: 0.911 → 0.737
- test boundary variance: 0.195 → 4.405
- validation/test direction cosine at end: 0.677
- class movement cosine at end: 0.577
- update-data coverage at end: 1.000

## 최소 원리

gradient training은 각 batch의 cross-entropy를 통해 output layer와 모든 hidden parameter를 동시에 업데이트하고, 서로 다른 class/sample의 gradient가 반복적으로 합쳐진다. inverse는 validation objective와 선택한 보존 제약을 만족하는 국소 해를 최소 norm으로 찾으므로 목표에는 직접적이지만, unseen distribution의 covariance와 모든 class loss를 동시에 학습한 것은 아니다.

`random representation → data-averaged gradient updates → shared representation geometry → validation/test alignment → generalization`

상세 update trajectory와 seed별 결과는 JSON에 저장했다.
