# Validation-only selective hidden geometry inverse

train의 80%로 모델을 학습하고 validation에서만 취약 sample과 safe anchor를 선택했다. test는 모든 update가 끝난 뒤 최종 평가에만 사용했다.

## Validation partition

평균 misclassified=23.6, vulnerable=30.4, safe=1443.6
validation accuracy: baseline=0.984 → selective=0.990

## Test 결과
baseline: accuracy=0.938, q10 margin=2.958, min margin=-26.617
- selective_inverse: accuracy=0.937, misclassified=185.8, q10 margin=2.939, min margin=-25.408, safe accuracy=1.000, safe validation margin change=0.302, gate change=0.8%
- full_margin_inverse: accuracy=0.938, misclassified=182.8, q10 margin=3.002, min margin=-26.952, safe accuracy=1.000, safe validation margin change=0.421, gate change=0.5%
- random_same_norm: accuracy=0.938, misclassified=181.6, q10 margin=2.950, min margin=-26.614, safe accuracy=1.000, safe validation margin change=0.047, gate change=0.1%
- gradient_same_norm: accuracy=0.914, misclassified=253.0, q10 margin=1.041, min margin=-29.030, safe accuracy=0.968, safe validation margin change=2.717, gate change=2.0%

## Repeated selective inverse
- accuracy: 0.938, 0.938, 0.938, 0.937, 0.937
- q10 margin: 2.965, 2.960, 2.954, 2.953, 2.949
- safe accuracy: 1.000, 1.000, 1.000, 1.000, 1.000

## 결론

validation에서는 선택 inverse가 개선되지만 test에서는 baseline보다 정확도가 낮고 오분류가 늘었다. 따라서 safe sample 보존 제약은 일반화됐지만, validation sample별 수정 방향은 test distribution 전체의 공통 경계 방향으로 일반화되지 않았다. confusion matrix와 seed별 원자료는 JSON에 저장했다.
