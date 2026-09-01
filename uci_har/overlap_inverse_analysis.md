# Boundary-overlap inverse design

SITTING/STANDING의 hidden2 분포와 output decision 방향을 비교하고, centroid를 보존하면서 boundary overlap만 줄이는 inverse를 validation에서 설계했다. test는 최종 평가에서만 사용했다.

## Baseline distribution
- accuracy=0.938, pair recall=0.917, pair confusion=82.0, boundary overlap=0.151, centroid distance=11.401, q10 margin=2.958
- pair sample 중 ambiguity 영역 비율=8.9%
- ambiguity 영역 오분류율=39.3%, 바깥 오분류율=5.3%

## Test comparison
- overlap_inverse: accuracy=0.930, pair recall=0.901, pair confusion=98.4, overlap=0.184, centroid distance=11.378, q10 margin=2.387, other preservation=0.988, gate change=3.6%
- centroid_inverse_same_norm: accuracy=0.892, pair recall=0.780, pair confusion=84.2, overlap=0.154, centroid distance=18.900, q10 margin=-0.943, other preservation=0.984, gate change=7.2%
- random_same_norm: accuracy=0.938, pair recall=0.918, pair confusion=80.8, overlap=0.152, centroid distance=11.382, q10 margin=2.933, other preservation=0.995, gate change=0.8%

## 지표-오류 연결
- centroid distance correlation=-0.203
- mean pair margin correlation=-0.429
- boundary overlap correlation=0.309

## 반복 overlap inverse
- accuracy: 0.937, 0.937, 0.937, 0.937, 0.937
- pair recall: 0.914, 0.916, 0.916, 0.916, 0.916
- overlap: 0.153, 0.154, 0.155, 0.155, 0.155

## 결론

baseline에서는 오류가 boundary overlap 영역에 집중됐다. 그러나 validation에서 설계한 overlap inverse는 test overlap을 0.151에서 0.184로 오히려 늘렸고 pair recall도 낮췄다. class distribution의 공통 방향은 오류 설명에는 유효하지만, validation sample의 overlap 제거 방향이 test distribution의 covariance·tail 구조까지 일반화되지는 않았다.
