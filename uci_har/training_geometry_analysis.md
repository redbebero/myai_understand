# 학습 중 class-separating geometry 형성

## 1

- 가설: 성능 향상은 hidden representation의 class-separating geometry 형성과 함께 일어난다.
- 실험: epoch 0,1,2,5,10,20,80에서 input·hidden1·hidden2의 거리와 분리도를 측정한다.
- 실제 결과: `"checkpoint별 원자료는 runs에 저장된다."`
- 모순: checkpoint 원자료에서 가설과 다른 seed·epoch를 확인한다.
- 수정된 원리: geometry와 accuracy의 상대적 onset 및 subspace 개입 효과로 판단한다.

## 2

- 가설: geometry가 accuracy보다 먼저 형성된다.
- 실험: 각 seed에서 초기값 대비 최종 변화의 10%에 도달하는 epoch를 accuracy와 hidden2 separation ratio에 대해 비교한다.
- 실제 결과: `[{"seed": 7, "accuracy": 1, "hidden2_geometry": 1}, {"seed": 11, "accuracy": 1, "hidden2_geometry": 1}, {"seed": 19, "accuracy": 1, "hidden2_geometry": 1}, {"seed": 23, "accuracy": 1, "hidden2_geometry": 1}, {"seed": 31, "accuracy": 1, "hidden2_geometry": 1}]`
- 모순: checkpoint 원자료에서 가설과 다른 seed·epoch를 확인한다.
- 수정된 원리: geometry와 accuracy의 상대적 onset 및 subspace 개입 효과로 판단한다.

## 3

- 가설: 초기 geometry도 이미 실제 판단에 사용된다.
- 실험: 각 checkpoint에서 class-separating hidden2 subspace와 random subspace를 제거해 test accuracy를 비교한다.
- 실제 결과: `"각 checkpoint의 class/random subspace accuracy가 runs에 저장된다."`
- 모순: checkpoint 원자료에서 가설과 다른 seed·epoch를 확인한다.
- 수정된 원리: geometry와 accuracy의 상대적 onset 및 subspace 개입 효과로 판단한다.

## 4

- 가설: 같은 발달 순서가 seed마다 반복된다.
- 실험: 5개 seed의 checkpoint 곡선과 onset epoch를 비교한다.
- 실제 결과: `"seed별 결과와 예외를 곡선 원자료로 비교한다."`
- 모순: checkpoint 원자료에서 가설과 다른 seed·epoch를 확인한다.
- 수정된 원리: geometry와 accuracy의 상대적 onset 및 subspace 개입 효과로 판단한다.

## 최종 판정

각 seed의 checkpoint 원자료를 기준으로 geometry 선행·동시 발달·후행·seed 의존성 중 하나를 선택한다.

