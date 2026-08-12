import json
from pathlib import Path


def generate_dataset(path="xor_dataset.json"):
    rows = [
        {"inputs": [0, 0], "target": 0},
        {"inputs": [0, 1], "target": 1},
        {"inputs": [1, 0], "target": 1},
        {"inputs": [1, 1], "target": 0},
    ]
    Path(path).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return rows


if __name__ == "__main__":
    generate_dataset()
    print("Wrote xor_dataset.json")
