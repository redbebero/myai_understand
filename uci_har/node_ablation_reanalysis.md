# Hidden2 노드 22·24 제거 재분석

## 질문

기존 test set에서 개별 ablation 영향이 가장 작았던 hidden2 노드 22와 24를 동시에 제거했을 때 관찰된 `+0.441%p` 정확도 상승이 재현 가능한 일반화 효과인지, 아니면 표본 변동·노드 중복·특정 클래스 보정 때문인지 평가했다.

## 원래 관찰

기존 실행은 test label로 개별 노드 중요도를 계산한 뒤 test에서 노드를 선택했다. 따라서 원래 `94.367% → 94.808%` 결과는 탐색적 결과이며 confirmatory 성능 비교가 아니다.

## 독립 재현 설계

- 모델: `561→64→32→6` ReLU MLP, Adam
- seeds: 7, 11, 19, 23, 31
- UCI 공식 test subjects는 최종 평가에만 사용
- 공식 train subjects 중 1, 3, 5, 6, 7, 8을 validation subjects로 고정
- validation에서 개별 hidden2 ablation accuracy drop이 가장 작은 두 노드를 선택
- 선택된 노드는 test를 본 뒤 바꾸지 않음
- test 평가는 원본과 두 노드 제거 모델의 paired prediction으로 비교

## 결과

| seed | validation 선택 노드 | validation 변화 | test 변화 |
|---:|---|---:|---:|
| 7 | 13, 1 | -0.47%p | -1.15%p |
| 11 | 22, 26 | +1.26%p | -0.24%p |
| 19 | 9, 13 | +1.05%p | +0.61%p |
| 23 | 28, 10 | +1.21%p | +0.24%p |
| 31 | 2, 5 | +0.42%p | -1.15%p |

다중 seed test 변화:

- 평균: **-0.339%p**
- 중앙값: **-0.238%p**
- 양의 변화: 5개 중 2개
- bootstrap 95% CI: **-0.970%p ~ +0.292%p**

따라서 test 독립적인 선택 절차에서는 두 노드 제거의 성능 개선이 재현되지 않았다.

## 기존 22·24 pair의 paired 분석

seed 7 full-train 모델에서 기존 pair를 직접 재분석했다.

- 개선: 40개
- 악화: 27개
- 변화 없음: 2,879개
- 정확도 변화: +0.441%p
- McNemar exact 양측 p: 0.142
- 샘플 bootstrap 95% CI: -0.102%p ~ +0.984%p

이는 방향성의 증거가 아니라, 작고 불확실한 순변화다.

## 클래스별 패턴

22·24 제거의 개선은 균등하지 않았다.

- walking: 개선 0, 악화 17
- walking_upstairs: 개선 19, 악화 0
- walking_downstairs: 개선 9, 악화 0
- sitting: 개선 8, 악화 0
- standing: 개선 0, 악화 10
- laying: 개선 4, 악화 0

전체 상승은 주로 upstairs/downstairs/sitting/laying 개선이 walking/standing 악화를 상쇄한 결과다.

## subject별 패턴

개선은 subject 9와 10에서 가장 컸고, subject 20에서는 10개가 악화되고 개선은 없었다. 효과는 subject-independent하지 않다.

## 노드 기능 단서

노드 22:

- 활성 비율: 92.0%
- 평균 activation: 5.40
- activation correlation with node 24: 0.323
- downstream weight cosine with node 24: 0.728

틀린 원본 샘플에서 노드 22의 평균 class contribution 절댓값이 더 컸다. 이는 일부 오답에서 과도한 logit 보정 신호였을 가능성과 일치하지만, walking/standing에서 제거 후 악화되어 전반적 유해 노드라고 할 수 없다.

노드 24:

- 활성 비율: 33.3%
- 평균 activation: 1.44

노드 24는 노드 22보다 덜 활성화되고 기여도도 작다. 중복 기능 또는 조건부 기능 가능성은 있지만, 이번 결과만으로 무의미한 노드라고 결론 낼 수 없다.

## 결론

현재 가장 방어 가능한 결론은 다음이다.

> Test label을 사용해 사후적으로 선택한 22·24 pair에서는 정확도가 0.441%p 상승했지만, paired 검정과 bootstrap 구간은 0 효과와 양립한다. Test-independent validation 선택을 5개 seed에서 반복했을 때 평균 test 변화는 -0.339%p였고 양의 변화는 2/5 seed뿐이었다. 따라서 관찰된 상승은 재현 가능한 일반화 개선으로 확인되지 않았다. 22·24는 특정 활동과 subject에서 서로 다른 방향의 logit 보정을 수행했을 가능성이 있으며, 중복성·과적합·경계 이동을 구분하려면 더 많은 seed와 validation 기반 pair 분석이 필요하다.

## 재현

```bash
python -m uci_har.node_ablation_reanalysis
```

결과:

```text
uci_har/node_ablation_reanalysis_results.json
```
