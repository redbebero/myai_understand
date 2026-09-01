# Sample-gradient averaging and generalization

동일 초기화에서 sample별 cross-entropy gradient를 계산하고, batch 평균 방향과 평균에서 벗어난 residual을 비교했다. test는 update에 사용하지 않고 평가만 했다.

## Batch-size summary
- batch 1: alignment=1.000, cancellation=0.000, test accuracy=0.359, test separation=1.004, val/test geometry cosine=0.983
- batch 4: alignment=0.488, cancellation=0.445, test accuracy=0.423, test separation=1.215, val/test geometry cosine=0.987
- batch 16: alignment=0.309, cancellation=0.648, test accuracy=0.548, test separation=1.584, val/test geometry cosine=0.992
- batch 64: alignment=0.265, cancellation=0.704, test accuracy=0.640, test separation=1.894, val/test geometry cosine=0.993
- batch 128: alignment=0.249, cancellation=0.718, test accuracy=0.670, test separation=1.994, val/test geometry cosine=0.994

## 최소 원리

평균 gradient는 sample별 요구의 합의된 방향이지만, 평균만으로 일반화가 보장되지는 않는다. residual은 평균에서 상쇄되며, batch가 커질수록 그 상쇄가 커지는지와 그 결과 geometry/test 성능이 함께 안정화되는지를 원자료로 판단한다.

`sample gradients → common/residual decomposition → averaging → distribution geometry → validation/test sharing → generalization`
