# W1/W2 기여의 크기·증폭·방향 분해

## 1

- 가설: W1의 우세는 단순히 W1 parameter update norm이 더 크기 때문이다.
- 통제 실험: 실제 W1/W2 group norm과 동일 norm counterfactual을 비교한다.
- 실제 결과: `{"W1": 0.11570514471527168, "W2": 0.030326349833089714}`
- W1 우세를 설명하는 정도: 원자료의 크기·증폭·동일 norm 방향 효과를 분리해 판단한다.
- 수정된 최소 메커니즘: parameter update가 Jacobian을 통해 hidden2 geometry로 전달되는 효율로 설명한다.

## 2

- 가설: W1 변화가 downstream W2를 거치며 hidden2에서 더 크게 증폭된다.
- 통제 실험: Jacobian predicted hidden norm / parameter norm과 distance gap을 비교한다.
- 실제 결과: `{"W1": {"amplification": 8.008795618446127, "predicted_hidden_norm": 0.9553536213955072}, "W2": {"amplification": 3.057631688628724, "predicted_hidden_norm": 0.09049808879688212}}`
- W1 우세를 설명하는 정도: 원자료의 크기·증폭·동일 norm 방향 효과를 분리해 판단한다.
- 수정된 최소 메커니즘: parameter update가 Jacobian을 통해 hidden2 geometry로 전달되는 효율로 설명한다.

## 3

- 가설: 동일 norm에서도 W1이 class-separating 방향으로 더 효율적이다.
- 통제 실험: W1/W2를 같은 unit norm으로 적용한 counterfactual의 distance gap을 비교한다.
- 실제 결과: `{"W1": 4.0901950664076825, "W2": 0.8497945294924577}`
- W1 우세를 설명하는 정도: 원자료의 크기·증폭·동일 norm 방향 효과를 분리해 판단한다.
- 수정된 최소 메커니즘: parameter update가 Jacobian을 통해 hidden2 geometry로 전달되는 효율로 설명한다.

## 4

- 가설: 이 세 효과의 조합이 seed마다 반복된다.
- 통제 실험: 5개 seed × 초기 10개 update의 group metric을 비교한다.
- 실제 결과: `"5개 seed 모두 동일 norm counterfactual과 실제 update metric을 JSON에 보존"`
- W1 우세를 설명하는 정도: 원자료의 크기·증폭·동일 norm 방향 효과를 분리해 판단한다.
- 수정된 최소 메커니즘: parameter update가 Jacobian을 통해 hidden2 geometry로 전달되는 효율로 설명한다.

## 최소 메커니즘

`W1 update → W2를 통한 Jacobian 전달 → 큰 hidden movement → between-class gap 증가`
