# UCI HAR Gradient Geometry 탐구

이 폴더는 UCI HAR `561→64→32→6` MLP에서 내부 representation geometry가 어떻게 형성되는지 탐구한 과정을 공유하기 위한 패키지다.

## 핵심 결론

일반화 가능한 representation은 특정 뉴런이나 한 번 계산한 geometry를 직접 지정해서 만들어지지 않았다. 현재 representation의 오차를 측정하고, 그 상태에서 gradient를 다시 계산하는 iterative feedback 과정에서 형성됐다.

동일 초기화·동일 128개 sample 비교:

| 방법 | Test accuracy | Test loss | 동일 parameter norm 후 accuracy |
|---|---:|---:|---:|
| One-shot 평균 gradient | 0.279 | 1.846 | 0.356 |
| Frozen gradient sequential | 0.338 | 9.544 | 0.342 |
| Recomputed sequential | 0.706 | 0.768 | 0.698 |

## 이 탐구에서 말하는 geometry

hidden2의 각 sample 표현을 좌표로 보고, 같은 class sample들이 얼마나 모여 있는지와 다른 class 중심들이 얼마나 떨어져 있는지를 측정했다.

- **class separation**: class 중심 사이의 분산을 class 내부 분산으로 나눈 값이다. 클수록 class가 더 잘 분리된다.
- **distribution overlap**: SITTING/STANDING의 decision 방향에서 두 class가 겹치는 정도다. 작을수록 경계 근처 혼동이 적다.
- **validation/test geometry cosine**: validation과 test에서 class 중심 구조가 같은 방향을 가지는지 보는 값이다. 단, 이 값 하나만으로 분류 성능을 판단하지 않는다.
- **hidden movement**: 초기 모델 대비 hidden2 표현이 얼마나 이동했는지 나타낸다.

geometry가 좋아진다는 말은 단순히 class 중심을 멀리 옮긴다는 뜻이 아니다. loss, decision boundary, class covariance가 함께 안정되는지를 의미한다.

## 데이터와 통제

UCI HAR의 원래 train 데이터를 다시 train/validation으로 나누고, 원래 test set은 끝까지 update에 사용하지 않았다. 표준화 평균과 scale도 train split에서만 계산했다.

모든 비교에서 다음을 고정했다.

- 입력 feature: 원본 561개
- 구조: `561→64→32→6`
- optimizer: Adam
- seed: `7, 11, 19, 23, 31`
- feature engineering 및 neuron search: 없음

따라서 test 결과는 update에 사용하지 않은 unseen sample에 대한 최종 평가다.

## 읽는 순서

1. [experiment_process.md](experiment_process.md): 가설이 어떻게 수정됐는지에 대한 전체 과정
2. [final_result.md](final_result.md): 마지막 결정 실험과 최종 원리
3. `results/gradient_feedback_results.json`: 마지막 실험의 seed별 원자료
4. `uci_har/gradient_feedback_experiment.py`: 마지막 실험 코드

## 실험 파일의 역할

- `gradient_vs_inverse_experiment.py`: validation geometry inverse와 gradient training 비교
- `gradient_averaging_experiment.py`: sample gradient의 평균 성분과 residual 성분 비교
- `gradient_averaging_controls.py`: 총 sample exposure와 sample 집합을 통제한 비교
- `gradient_feedback_experiment.py`: one-shot, frozen, recomputed gradient의 최종 결정 실험

각 JSON은 평균값만이 아니라 seed별 trajectory도 포함한다. 따라서 평균 결과가 특정 seed 하나에 의해 만들어졌는지 확인할 수 있다.

## 재현

원본 UCI HAR Dataset을 다운로드해 다음 위치에 둔다.

```text
요약한폴더/uci_har/UCI HAR Dataset/
```

그 다음 공유 폴더에서 실행한다.

```bash
cd 요약한폴더
python -m uci_har.gradient_feedback_experiment
```

공유 폴더의 `uci_har/`에는 실행에 필요한 기존 분석 모듈과 테스트 코드가 포함되어 있다. 데이터셋 원본은 크기 때문에 포함하지 않았다.

## 실험 조건

- UCI HAR 원본 입력 561개
- MLP: `561→64→32→6`
- strict train/validation/test 분리
- 5개 seed: `7, 11, 19, 23, 31`
- Adam cross-entropy
- test는 update 설계에 사용하지 않고 최종 평가에만 사용
- 새 feature/neuron 탐색 없음
