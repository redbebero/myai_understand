import json
import math
from pathlib import Path


FORMULA = json.loads((Path(__file__).parent / "recovered_formula.json").read_text(encoding="utf-8"))["formula"]


def predict(inputs):
    x, y = inputs
    radius = math.hypot(x, y)
    angle = math.atan2(y, x)
    score = FORMULA["polarity"] * math.cos(angle - FORMULA["frequency"] * radius + FORMULA["phase"])
    return 1 / (1 + math.exp(-FORMULA["scale"] * score))
