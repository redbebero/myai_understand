# Validation class-pair decision subspace inverse

validation confusion에서 반복 혼동 pair를 찾고, 해당 pair의 hidden mean direction과 output decision direction을 함께 제약했다. test는 모든 update 후 최종 평가했다.

pair frequency: {'(3, 4)': 5}
baseline test: accuracy=0.938, pair recall=0.917, pair margin=13.637, subspace separation=2.658

## Test 결과
- sample_selective_inverse: accuracy=0.937, pair recall=0.916, pair margin=13.685, subspace separation=2.679, other accuracy=0.948, other preservation=0.998, gate change=0.8%
- class_pair_inverse: accuracy=0.938, pair recall=0.916, pair margin=15.003, subspace separation=2.703, other accuracy=0.950, other preservation=0.998, gate change=0.9%
- random_same_norm: accuracy=0.938, pair recall=0.917, pair margin=13.641, subspace separation=2.658, other accuracy=0.950, other preservation=1.000, gate change=0.1%
- gradient_same_norm: accuracy=0.925, pair recall=0.881, pair margin=13.272, subspace separation=2.567, other accuracy=0.949, other preservation=0.994, gate change=1.4%

## 반복 pair inverse
- accuracy: 0.938, 0.938, 0.938, 0.938, 0.938
- pair recall: 0.917, 0.917, 0.917, 0.917, 0.917
- subspace separation: 2.668, 2.678, 2.688, 2.697, 2.705

## 해석

class-level inverse는 pair margin과 hidden subspace separation은 증가시켰지만, discrete pair recall은 baseline보다 높아지지 않았다. 따라서 class-level geometry의 일부는 test에 일반화됐지만, decision threshold를 넘는 새로운 정답 예측으로 이어지지는 않았다.
