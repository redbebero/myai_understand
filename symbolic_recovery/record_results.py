import json
from pathlib import Path

from evaluate import evaluate


ROOT = Path(__file__).parent


def main():
    result = json.loads((ROOT / "recovered_formula.json").read_text(encoding="utf-8"))
    labeled = evaluate(ROOT.parent / "two_spiral" / "spiral_test.json")
    lines = [
        "# 수식 복원 실험 결과",
        "",
        "## 질문",
        "",
        "학습 데이터와 라벨을 보지 않고, 모델 구조와 가중치만으로 더 단순한 수식을 찾을 수 있는가?",
        "",
        "## 제한 조건 확인",
        "",
        "수식 탐색은 `target_model.json`과 무작위 입력에 대한 모델 출력만 사용했다.",
        f"라벨 사용 여부: {result['labels_used_during_recovery']}",
        f"탐색 입력 수: {result['search_probe_count']}",
        f"검증 입력 수: {result['verification_probe_count']}",
        "",
        "## 발견한 수식",
        "",
        "```text",
        "r = hypot(x, y)",
        "theta = atan2(y, x)",
        "q = sigmoid(8 * polarity * cos(theta - frequency * r + phase))",
        "```",
        "",
        f"frequency = {result['formula']['frequency']}",
        f"phase = {result['formula']['phase']}",
        f"polarity = {result['formula']['polarity']}",
        f"탐색 입력에서 모델과 일치율: {result['search_agreement']:.1%}",
        f"숨겨둔 입력에서 모델과 일치율: {result['verification_agreement']:.1%}",
        "",
        "## 라벨을 사용한 마지막 검증",
        "",
        f"기존 모델 정확도: {labeled['model_accuracy']:.1%}",
        f"복원 수식 정확도: {labeled['formula_accuracy']:.1%}",
        f"기존 모델과 수식의 분류 일치율: {labeled['model_formula_agreement']:.1%}",
        "",
        "## 결론",
        "",
        "가중치만으로 모델의 계산을 완전히 같은 하나의 수식으로 바꾸지는 못했다. 일반적인 극좌표 삼각함수 후보를 사용하면 모델의 일부 결정 경계를 짧은 코드로 근사할 수 있지만, 일치율이 낮다면 이 모델의 계산이 단순한 하나의 주기식으로 표현되지 않는다는 뜻이다.",
        "",
        "이 결과는 실패가 아니라 질문을 구체화한다. 수식 복원을 위해서는 가중치만이 아니라 입력 분포, 뉴런 활성화, 후보 함수의 선택이 추가로 필요할 수 있다.",
    ]
    (ROOT / "recovery_record.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ROOT / "labeled_evaluation.json").write_text(json.dumps(labeled, indent=2) + "\n", encoding="utf-8")
    print("Wrote recovery_record.md and labeled_evaluation.json")


if __name__ == "__main__":
    main()
