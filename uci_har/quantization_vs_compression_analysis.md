# 양자화와 hidden 표현 압축 비교

양자화는 같은 MLP 가중치를 낮은 bit로 저장한 뒤 복원해 평가했다. 표현 압축은 기존 실험 결과의 frozen output-layer 방식이며, 원래 MLP를 제거하지 않고 hidden 표현만 투영·복원한다.

| 방식 | 정확도 | 저장 크기 | hidden 활성값 |
|---|---:|---:|---:|
| FP32 원본 | 0.938 | 149.4 KB | 128 B |
| INT8 양자화 | 0.939 | 37.3 KB | 128 B |
| INT4 양자화 | 0.928 | 18.7 KB | 128 B |
| INT2 양자화 | 0.305 | 9.3 KB | 128 B |
| pca k=8 | 0.938 | 150.5 KB* | 32 B |
| class_separating k=8 | 0.937 | 150.5 KB* | 32 B |
| supervised_output k=8 | 0.938 | 150.5 KB* | 32 B |

* 현재 구현은 원래 MLP를 유지하고 projection basis와 평균을 추가하므로 모델 저장 크기와 원래 dense 연산을 줄이지 않는다. 줄어드는 것은 hidden activation 저장량이다.

원래 dense MACs: 38,144. k=8 projection/reconstruction 추가 MACs: 512.

## 결론

같은 정확도에 가까운 조건에서 INT8은 모델 저장 크기를 약 4분의 1로 줄이는 반면, 현재의 표현 압축은 k=8 hidden activation을 4분의 1로 줄이지만 모델 자체는 작아지지 않는다. 실제 모델 크기와 연산량까지 줄이려면 projection을 학습 그래프 안에 넣고 앞뒤 weight를 구조적으로 다시 접어야 한다.
