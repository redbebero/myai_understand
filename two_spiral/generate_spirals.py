import json
import math
import random
from pathlib import Path


def generate(path, points_per_class=60, noise=0.08, seed=7):
    rng = random.Random(seed)
    rows = []
    for label in (0, 1):
        for index in range(points_per_class):
            radius = index / points_per_class
            angle = 4 * math.pi * radius + label * math.pi
            angle += rng.uniform(-noise, noise)
            rows.append({
                "inputs": [
                    radius * math.cos(angle) + rng.uniform(-noise, noise),
                    radius * math.sin(angle) + rng.uniform(-noise, noise),
                ],
                "target": label,
            })
    rng.shuffle(rows)
    Path(path).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return rows


if __name__ == "__main__":
    root = Path(__file__).parent
    generate(root / "spiral_train.json", seed=7)
    generate(root / "spiral_test.json", seed=19)
    print("Wrote spiral_train.json and spiral_test.json")
