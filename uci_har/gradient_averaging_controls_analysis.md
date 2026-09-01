# Exposure-matched gradient averaging controls

실험 A는 모든 batch size가 같은 1280개 train sample을 보게 했고, 실험 B는 같은 128개 sample을 batch 순서만 다르게 처리했다. test는 update에 사용하지 않았다.

## Experiment A: same sample exposure
- batch 1: updates=1280, movement=5.084, cancellation=-0.000, test accuracy=0.823, test separation=3.415, val/test cosine=0.993
- batch 4: updates=320, movement=3.095, cancellation=0.308, test accuracy=0.864, test separation=3.269, val/test cosine=0.993
- batch 16: updates=80, movement=2.199, cancellation=0.637, test accuracy=0.880, test separation=3.064, val/test cosine=0.993
- batch 64: updates=20, movement=1.508, cancellation=0.803, test accuracy=0.786, test separation=2.561, val/test cosine=0.994
- batch 128: updates=10, movement=1.066, cancellation=0.814, test accuracy=0.670, test separation=1.994, val/test cosine=0.994

## Experiment B: same 128-sample set
- batch 1: updates=128, movement=2.016, cancellation=-0.000, test accuracy=0.706, test separation=2.419, val/test cosine=0.993, same-norm test accuracy=0.586, same-norm separation=1.316
- batch 4: updates=32, movement=1.306, cancellation=0.418, test accuracy=0.626, test separation=2.021, val/test cosine=0.993, same-norm test accuracy=0.587, same-norm separation=1.518
- batch 16: updates=8, movement=0.783, cancellation=0.674, test accuracy=0.496, test separation=1.394, val/test cosine=0.991, same-norm test accuracy=0.497, same-norm separation=1.614
- batch 64: updates=2, movement=0.336, cancellation=0.731, test accuracy=0.356, test separation=0.669, val/test cosine=0.982, same-norm test accuracy=0.400, same-norm separation=1.657
- batch 128: updates=1, movement=0.195, cancellation=0.718, test accuracy=0.279, test separation=0.511, val/test cosine=0.976, same-norm test accuracy=0.381, same-norm separation=1.652

## 판정

Experiment A에서 batch 차이가 남으면 exposure만으로 설명되지 않는다. Experiment B에서 같은 sample set에서도 동시 averaging이 더 안정적인 geometry를 만들면 averaging 가설을 지지하지만, Adam update 횟수와 parameter movement가 다르므로 그 차이를 함께 해석해야 한다.

`same data → averaging schedule → residual cancellation → representation geometry → unseen generalization`
