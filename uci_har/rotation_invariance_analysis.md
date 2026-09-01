# Adam 좌표계 의존성

## 1

- 가설: 같은 정보를 회전해도 raw gradient와 SGD 기능 방향은 보존된다.
- 통제 실험: 회전된 gradient/update를 원래 좌표로 되돌려 cosine과 geometry를 비교한다.
- 실제 결과: `{"raw_gradient": {"update_cosine": 1.0, "original_alignment": 0.8933221260890971, "rotated_alignment": 0.893322126089097, "original_geometry_gain": 3.4528600091675314, "rotated_geometry_gain": 3.452860009167531}, "sgd": {"update_cosine": 1.0, "original_alignment": 0.8933221260890964, "rotated_alignment": 0.8933221260890966, "original_geometry_gain": 3.4528600091675314, "rotated_geometry_gain": 3.452860009167531}}`
- 맞지 않는 점: raw/SGD와 Adam/scalar-v의 basis 의존성을 분리한다.
- 수정된 원리: Adam의 elementwise second-moment scaling이 회전에서 보존되지 않는 방향 변환을 만든다.

## 2

- 가설: coordinate-wise Adam은 rotation invariant하지 않다.
- 통제 실험: 원래 basis Adam update를 회전한 값과 회전 basis에서 직접 계산한 Adam update를 비교한다.
- 실제 결과: `{"adam": {"update_cosine": 0.8029504478564607, "original_alignment": 0.8846722438187494, "rotated_alignment": 0.8183878298884703, "original_geometry_gain": 4.092726547039819, "rotated_geometry_gain": 3.4310057835590997}}`
- 맞지 않는 점: raw/SGD와 Adam/scalar-v의 basis 의존성을 분리한다.
- 수정된 원리: Adam의 elementwise second-moment scaling이 회전에서 보존되지 않는 방향 변환을 만든다.

## 3

- 가설: scalar-v preconditioning은 basis 의존성을 줄인다.
- 통제 실험: scalar-v와 full Adam의 update cosine·alignment·geometry gain을 비교한다.
- 실제 결과: `{"scalar_v": {"update_cosine": 1.0, "original_alignment": 0.96202358131858, "rotated_alignment": 0.96202358131858, "original_geometry_gain": 4.755992482610675, "rotated_geometry_gain": 4.755992482610675}, "full_adam": {"update_cosine": 0.8029504478564607, "original_alignment": 0.8846722438187494, "rotated_alignment": 0.8183878298884703, "original_geometry_gain": 4.092726547039819, "rotated_geometry_gain": 3.4310057835590997}}`
- 맞지 않는 점: raw/SGD와 Adam/scalar-v의 basis 의존성을 분리한다.
- 수정된 원리: Adam의 elementwise second-moment scaling이 회전에서 보존되지 않는 방향 변환을 만든다.

## 최소 원리

`같은 정보 → basis rotation → gradient/SGD 기능 보존 → elementwise v_t scaling 방향 변화 → hidden geometry 차이`
