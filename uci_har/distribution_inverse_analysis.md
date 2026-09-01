# Distribution-level covariance inverse

validation 전체 class의 boundary 방향 variance를 30% 줄이고 SITTING/STANDING centroid hidden vector를 보존하는 distribution-level inverse를 계산했다. test는 최종 평가에만 사용했다.

validation overlap: baseline=0.062 → distribution inverse=0.044

## Test 결과
- baseline: accuracy=0.938, boundary variance=(11.244,7.535), overlap=0.151, centroid distance=11.401, q10 margin=2.958, other preservation=1.000, gate change=0.0%
- distribution_inverse: accuracy=0.938, boundary variance=(10.653,7.128), overlap=0.143, centroid distance=11.353, q10 margin=2.888, other preservation=0.993, gate change=1.9%
- sample_overlap_inverse_same_norm: accuracy=0.937, boundary variance=(11.387,7.793), overlap=0.155, centroid distance=11.441, q10 margin=2.954, other preservation=0.995, gate change=1.3%
- centroid_inverse_same_norm: accuracy=0.939, boundary variance=(18.989,12.037), overlap=0.140, centroid distance=14.466, q10 margin=3.147, other preservation=0.993, gate change=2.9%
- random_same_norm: accuracy=0.938, boundary variance=(11.229,7.529), overlap=0.151, centroid distance=11.395, q10 margin=2.956, other preservation=0.999, gate change=0.3%

## 반복 distribution inverse
- accuracy: 0.938, 0.938, 0.938, 0.939, 0.938
- overlap: 0.147, 0.145, 0.143, 0.142, 0.141
- class-3 variance: 11.031, 10.885, 10.764, 10.680, 10.590

## 결론

distribution-level variance target은 validation에서 정의한 covariance 변화가 unseen test에서도 일부 재현됐다. overlap과 pair recall은 소폭 개선됐지만 전체 정확도와 하위 margin은 완전히 개선되지 않았으므로, distribution geometry는 오류를 결정하는 더 적절한 단위이지만 충분조건은 아니다.
