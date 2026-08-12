import copy
import json
import unittest
from pathlib import Path

from train_xor import evaluate, load_dataset, nonzero_parameters, prune_connection, train


ROOT = Path(__file__).parent
MODEL_COPY = ROOT / "xor_model.json"
RECORD = ROOT / "pruning_record.md"


def run_pruning_experiment():
    rows = load_dataset(ROOT / "xor_dataset.json")
    model = train(rows, epochs=10_000, seed=7)
    MODEL_COPY.parent.mkdir(exist_ok=True)
    MODEL_COPY.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")

    edges = [
        (0, 0, 0, "h1 <- x1"),
        (0, 0, 1, "h1 <- x2"),
        (0, 1, 0, "h2 <- x1"),
        (0, 1, 1, "h2 <- x2"),
        (1, 0, 0, "y <- h1"),
        (1, 0, 1, "y <- h2"),
    ]
    baseline = evaluate(model, rows)
    results = []
    for layer, row, column, name in edges:
        candidate = copy.deepcopy(model)
        prune_connection(candidate, layer, row, column)
        results.append((name, evaluate(candidate, rows)))

    lines = [
        "# XOR 연결 제거 기록",
        "",
        "기준 모델: 2개 입력 → 2개 은닉 뉴런 → 1개 출력 뉴런",
        f"기준 정확도: {baseline:.0%}",
        f"기준 연결 수: {nonzero_parameters(model)}개",
        "",
        "재학습 없이 연결 하나를 0으로 바꾼 결과:",
        "",
        "| 제거한 연결 | 남은 연결 수 | 정확도 |",
        "|---|---:|---:|",
    ]
    for name, accuracy in results:
        lines.append(f"| `{name}` | {nonzero_parameters(model) - 1} | {accuracy:.0%} |")
    lines += [
        "",
        "## 해석",
        "",
        "현재 학습 결과에서는 연결 하나만 제거해도 정확도가 100%에서 떨어졌다.",
        "따라서 이 모델의 현재 가중치 배치에서는 6개 연결이 모두 기능에 기여한다.",
        "이는 모든 XOR 모델에 연결 6개가 반드시 필요하다는 뜻은 아니다.",
        "다른 초기값으로 다시 학습하거나 제거 후 재학습하면 더 작은 구조가 나올 수 있다.",
    ]
    RECORD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return baseline, results


class PruningExperimentTest(unittest.TestCase):
    def test_pruning_record(self):
        baseline, results = run_pruning_experiment()
        self.assertEqual(baseline, 1.0)
        self.assertEqual(len(results), 6)
        self.assertTrue(MODEL_COPY.exists())
        self.assertTrue(RECORD.exists())
        self.assertTrue(all(accuracy < baseline for _, accuracy in results))


if __name__ == "__main__":
    unittest.main()
