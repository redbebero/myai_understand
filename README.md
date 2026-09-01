이 저장소에서 내가 수행한 AI 내부표현 분석을 이해하고 싶다.

먼저 다음 파일만 이 순서대로 읽어라.
1. plan.md
2. human_understandable_model_workflow.md
3. uci_har/reality_representation/PLAN.md
4. uci_har/reality_representation/experiment.py
5. uci_har/reality_representation/temporal_pattern_analysis.py
6. uci_har/reality_representation/REPORT.md
7. uci_har/reality_representation/TEMPORAL_REPORT.md

파일을 수정하지 말고, 읽은 내용만 설명하라.

다음 순서로 설명하라.

1. 데이터
   - 센서 채널이 실제로 몇 개인가?
   - 각 채널은 무엇을 측정하는가?
   - 128개 시간값은 무엇인가?
   - 1152라는 숫자는 어떻게 계산되는가?

2. 현재 모델
   - 실제 코드의 신경망 구조를 정확히 설명하라.
   - CNN인지 MLP인지 구분하라.
   - 64, 32, 6이 각각 무엇을 뜻하는가?
   - 학습에서 forward pass, loss, backpropagation이 어떻게 연결되는가?

3. 내부 표현
   - 32개 hidden 값이 무엇인지 설명하라.
   - 내부 방향이 신경망의 실제 층인지, 학습 후 분석을 위해 정의한 것인지 구분하라.
   - 마지막 분류층의 weight와 내부 방향이 어떤 관계인지 설명하라.

4. 내부 방향 분석
   - 코드에서 내부 방향을 어떤 계산으로 찾는지 설명하라.
   - 방향 성분을 어떻게 제거하는지 설명하라.
   - 제거 후 어떤 activity-pair margin을 측정하는지 설명하라.

5. 센서 정보 연결
   - 내부 방향과 원래 센서 feature를 어떻게 연결했는지 설명하라.
   - level, variation, energy, coupling의 의미를 설명하라.
   - correlation으로 말할 수 있는 것과 말할 수 없는 것을 구분하라.

6. 사람이 이해할 수 있는 추상화
   - walking vs walking_upstairs
   - sitting vs standing
   - walking_upstairs vs walking_downstairs
   각각에 대해:
   - 중요한 내부 방향
   - 연결된 센서 정보
   - 두 활동의 시간 패턴 차이
   - 사람이 이해할 수 있는 문장
   을 정리하라.

7. 검증과 한계
   - 관계를 파괴했을 때 정확도가 어떻게 변했는가?
   - intervention과 matched control 결과는 무엇인가?
   - 현재 무엇을 주장할 수 있는가?
   - 아직 무엇을 주장하면 안 되는가?

마지막에 다음 한 문장으로 전체 흐름을 요약하라.

"전체 raw sensor 입력 → AI 내부 표현 → 분류에 영향을 주는 내부 방향 → 원래 센서 패턴 → 사람이 이해할 수 있는 활동 정보"

전문용어를 쓰되, 각 전문용어를 바로 이어서 쉬운 말로 설명하라.
결과를 과장하지 말고, 코드와 보고서에 실제로 있는 내용만 사용하라.
'
