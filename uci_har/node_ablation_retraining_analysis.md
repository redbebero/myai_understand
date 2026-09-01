# Hidden2 노드 제거 후 재학습 분석

## 목적

낮은 중요도 노드 제거가 실제로 유해한 계산을 없애는 것인지, 단순히 기존 가중치를 깨뜨리는 것인지 구분했다.

## 설계

- UCI HAR `561→64→32→6` ReLU MLP
- seeds: 7, 11, 19, 23, 31
- 공식 train subjects 중 1, 3, 5, 6, 7, 8을 validation으로 사용
- validation에서 개별 ablation drop이 가장 작은 hidden2 노드 2개 선택
- test는 노드 선택에 사용하지 않음
- 세 조건 비교:
  1. baseline
  2. post-hoc: 선택된 노드의 출력·입력·출력 연결을 0으로 설정
  3. fine-tuned: post-hoc 모델을 fit subjects에서 30 epoch 추가 학습하며 제거 노드는 고정

## 결과

| seed | 선택 노드 | baseline test | post-hoc test | fine-tuned test |
|---:|---|---:|---:|---:|
| 7 | 13, 1 | 94.33% | 93.18% | 94.30% |
| 11 | 22, 26 | 94.57% | 94.33% | 94.81% |
| 19 | 9, 13 | 94.20% | 94.81% | 94.67% |
| 23 | 28, 10 | 94.30% | 94.54% | 94.64% |
| 31 | 2, 5 | 94.33% | 93.18% | 94.71% |

평균 test accuracy:

- baseline: **94.347%**
- post-hoc: **94.007%**
- fine-tuned: **94.625%**

평균 변화:

- post-hoc vs baseline: **-0.339%p**
- fine-tuned vs baseline: **+0.278%p**

## 해석

post-hoc 제거는 평균적으로 성능을 낮췄다. 따라서 낮은 개별 ablation drop이 곧 노드가 불필요하다는 뜻은 아니다. 노드는 다른 노드와 함께 계산에 참여하고 있으며, 제거 직후에는 기존 출력층이 그 변화에 맞춰 조정되지 않는다.

30 epoch fine-tuning 후에는 평균 정확도가 baseline보다 0.278%p 높아졌다. 이는 남은 노드와 출력층이 제거된 기능 일부를 보상할 수 있음을 보여준다. 다만 seed별 결과는 혼재했고, 이 실험만으로 일반적인 개선을 주장할 수 없다.

## 구조 결과

Test hidden representation의 centroid separation ratio는 post-hoc 제거에서 일부 seed에서 증가했지만 정확도와 일관되게 연결되지 않았다. 따라서 클래스 중심 간 거리가 커지는 것만으로 분류 성능 또는 표현의 질을 판단할 수 없다.

## 결론

가장 타당한 설명은 다음이다.

> 낮은 중요도 노드 2개는 독립적으로 보면 영향이 작지만, 기존 네트워크의 다른 계산과 결합되어 있다. 즉시 제거하면 일부 정보가 사라져 성능이 감소할 수 있고, 추가 재학습을 하면 남은 네트워크가 이를 부분적으로 보상할 수 있다. 원래 관찰된 22·24 제거의 소폭 상승은 일반적 pruning 효과라기보다 특정 seed·활동·subject에서의 경계 이동 또는 표본 변동으로 보는 것이 안전하다.

## 재현

```bash
python -m uci_har.node_ablation_retraining
```

결과:

```text
uci_har/node_ablation_retraining_results.json
```
