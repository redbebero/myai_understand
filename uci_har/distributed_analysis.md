# 분산 계산 패턴 분석

가설: 세부 활동 판단은 고정된 뉴런이나 pair가 아니라 여러 뉴런과 시간적 센서 특징의 분산 pattern으로 구현될 수 있다.

## seed_activation_patterns

- 가설: 같은 활동은 seed가 달라도 비슷한 내부 패턴을 만든다.
- 실험: 동일 test dynamic samples의 layer별 activation distance structure와 coactivation structure 비교
- 실제 결과: `{"1": {"distance_structure_similarity": 0.9625099884473342, "coactivation_structure_similarity": 0.6183505715970572, "top_unit_overlap": 0}, "2": {"distance_structure_similarity": 0.9440587709551659, "coactivation_structure_similarity": 0.647428797903923, "top_unit_overlap": 0}}`
- 기존 설명과 맞지 않는 점: 뉴런 ID overlap이 낮아도 distance structure가 높으면 분산 표현의 증거지만, 둘 다 낮으면 공통 표현 가설이 약해진다.
- 수정된 다음 질문: 공통 구조가 class별로 유지되는가?

## common_and_differential_activity

- 가설: 세 동적 활동은 공통 운동 표현과 활동별 차별 표현을 함께 가진다.
- 실험: class centroid distances, active-unit counts, contrast energy 비교
- 실제 결과: `[{"seed": 7, "layers": [{"layer": 1, "centroid_distances": {"0-1": 12.300845575286855, "0-2": 14.829012614884286, "1-2": 15.927390514287781}, "concentration": {"top_1_fraction": 0.11542578626424535, "top_2_fraction": 0.18194592843017615, "top_4_fraction": 0.3086403118030945, "top_8_fraction": 0.4885988265469742}}, {"layer": 2, "centroid_distances": {"0-1": 20.662958896503646, "0-2": 25.103774875503998, "1-2": 23.98958027884796}, "concentration": {"top_1_fraction": 0.15822912221895905, "top_2_fraction": 0.2758108051811145, "top_4_fraction": 0.4585573655895844, "top_8_fraction": 0.6710963178102204}}]}, {"seed": 11, "layers": [{"layer": 1, "centroid_distances": {"0-1": 10.820655161171976, "0-2": 12.58072146250192, "1-2": 12.977150244748874}, "concentration": {"top_1_fraction": 0.11789249686419877, "top_2_fraction": 0.23037852426745167, "top_4_fraction": 0.38732794117494324, "top_8_fraction": 0.5791270360718638}}, {"layer": 2, "centroid_distances": {"0-1": 16.441974534880284, "0-2": 19.491007375909437, "1-2": 16.162054946386164}, "concentration": {"top_1_fraction": 0.1890735650323529, "top_2_fraction": 0.28717378229307616, "top_4_fraction": 0.4638063216995196, "top_8_fraction": 0.6918405314563646}}]}]`
- 기존 설명과 맞지 않는 점: top-1 energy가 크면 분산 가설보다 소수 unit 가설이 강해진다.
- 수정된 다음 질문: 차별 표현은 어떤 sensor feature group과 연결되는가?

## feature_links

- 가설: 세부 표현은 entropy/frequency/jerk/autocorrelation feature와 연결된다.
- 실험: train dynamic subset에서 feature-activation correlation과 class effect 계산
- 실제 결과: `[{"seed": 7, "groups": {"entropy": 0.7843353703477485, "frequency": 0.37777087916428154, "jerk": 0.7803383627994376, "autocorrelation": 0.8664626446275737}, "top_features": ["tBodyAccMag-energy()", "tGravityAccMag-energy()", "fBodyAccMag-mean()", "fBodyAccMag-sma()", "fBodyAccMag-energy()", "tBodyAccMag-std()", "tGravityAccMag-std()", "tBodyAccMag-mad()"]}, {"seed": 11, "groups": {"entropy": 0.7458988656678747, "frequency": 0.36560027141605794, "jerk": 0.7346131283576779, "autocorrelation": 0.815628739139338}, "top_features": ["tBodyAccMag-energy()", "tGravityAccMag-energy()", "fBodyAccMag-energy()", "tBodyAccMag-std()", "tGravityAccMag-std()", "tBodyAccMag-mad()", "tGravityAccMag-mad()", "fBodyAccMag-mad()"]}]`
- 기존 설명과 맞지 않는 점: seed별 group ranking이 다르면 특정 feature family의 안정적 의미를 약하게 표현한다.
- 수정된 다음 질문: 연결된 feature/activation pattern을 교란하면 세부 분류가 무너지는가?

## intervention

- 가설: 분산 pattern과 연결된 feature/activation을 교란하면 random보다 세부 분류가 더 크게 변한다.
- 실험: dynamic test subset에서 선택된 units/features와 같은 크기 random control 교란
- 실제 결과: `[{"seed": 7, "intervention": {"baseline_accuracy": 0.9531362653208363, "selected_activation": {"units": [29, 9, 26, 1], "accuracy": 0.7786589762076424, "loss_change": 1.2601570448511061}, "random_activation": {"units": [28, 3, 19, 15], "accuracy": 0.9538572458543619, "loss_change": 0.003064925942598107}, "selected_features": {"indices": [206, 219, 502, 507, 508, 201, 214, 202], "accuracy": 0.9214131218457101, "loss_change": 0.1933627411439669}, "random_features": {"indices": [20, 42, 407, 236, 273, 301, 232, 91], "accuracy": 0.9524152847873107, "loss_change": -0.0035432349329052593}}}, {"seed": 11, "intervention": {"baseline_accuracy": 0.9610670511896179, "selected_activation": {"units": [8, 19, 5, 10], "accuracy": 0.6885364095169431, "loss_change": 1.7240088571594803}, "random_activation": {"units": [5, 13, 4, 22], "accuracy": 0.9495313626532084, "loss_change": 0.039103545636544196}, "selected_features": {"indices": [206, 219, 508, 201, 214, 202, 215, 504], "accuracy": 0.9315068493150684, "loss_change": 0.1382802726746446}, "random_features": {"indices": [60, 400, 79, 446, 426, 474, 364, 259], "accuracy": 0.9632299927901946, "loss_change": 0.0011831340240092691}}}]`
- 기존 설명과 맞지 않는 점: 선택 교란 효과가 random보다 작거나 seed마다 방향이 다르면 인과적 해석이 약해진다.
- 수정된 다음 질문: 서로 다른 unit 조합이 같은 feature-level 기능을 대체하는가?

## alternative_circuits

- 가설: seed마다 다른 unit이지만 비슷한 representation geometry와 feature group을 만든다.
- 실험: top-unit overlap과 permutation-invariant distance/coactivation similarity 비교
- 실제 결과: `{"1": {"distance_structure_similarity": 0.9625099884473342, "coactivation_structure_similarity": 0.6183505715970572, "top_unit_overlap": 0}, "2": {"distance_structure_similarity": 0.9440587709551659, "coactivation_structure_similarity": 0.647428797903923, "top_unit_overlap": 0}}`
- 기존 설명과 맞지 않는 점: geometry similarity가 낮으면 대체 회로 가설을 지지하지 않는다.
- 수정된 다음 질문: 더 많은 seed와 별도 모델 구조에서도 반복되는가?

## reframed_question

- 가설: 기능은 뉴런 번호가 아니라 계산 pattern으로 설명해야 한다.
- 실험: 모든 단계 결과를 neuron ID, pattern geometry, feature links, interventions로 분리 기록
- 실제 결과: `{"interpretation": "뉴런 번호의 재현성보다 pattern-level evidence를 우선한다."}`
- 기존 설명과 맞지 않는 점: 현재 두 seed만으로 분산 pattern의 보편성을 증명할 수 없다.
- 수정된 다음 질문: 더 많은 seed와 raw temporal model에서 같은 pattern이 재현되는가?

## 최종 해석

현재 결과는 특정 뉴런 번호를 기능의 이름으로 삼는 해석을 약화한다. 대신 seed 간 거리 구조·공동 활성화·feature 연결·교란 효과가 함께 재현되는지를 기능의 증거로 사용해야 한다. 이 증거가 약하면 분산 pattern 가설도 확정하지 않는다.

## 재현

`python -m uci_har.distributed_experiment`
