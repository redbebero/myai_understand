# 실제 8차원 bottleneck 비교

동일한 조건으로 원본 `561→64→32→6`과 bottleneck `561→64→8→6`을 각각 처음부터 학습했다. 이전 post-hoc projection과 달리 8차원 hidden layer 자체가 학습된다.

| 지표 | 원본 32D | bottleneck 8D | 변화율 |
|---|---:|---:|---:|
| Test accuracy | 0.938 | 0.946 | +0.83% |
| Cross-entropy | 0.369 | 0.286 | -22.60% |
| Parameters | 38246 | 36542 | -4.46% |
| FP32 model storage | 149.4 KB | 142.7 KB | -4.46% |
| FP32 activation memory/sample | 408 B | 312 B | -23.53% |
| Median inference time | 73.65 ms | 71.83 ms | -2.46% |
| MACs/sample | 38144 | 36464 | -4.40% |
| FLOPs/sample | 76288 | 72928 | -4.40% |

## 해석

이 결과는 post-hoc 표현 압축이 아니라 실제로 8차원 hidden layer를 학습한 결과다. 파라미터와 연산량 감소 폭은 첫 번째 561→64 층이 대부분의 비용을 차지하기 때문에 제한적이다. activation memory는 hidden 32차원에서 8차원으로 줄어든 만큼 감소한다. 추론 시간은 CPU·BLAS 환경의 측정값이므로 방향성 참고용이다.
