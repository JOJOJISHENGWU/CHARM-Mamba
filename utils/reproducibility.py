from __future__ import annotations

import json
from pathlib import Path


def load_json(path: str | Path) -> dict:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def assert_metric_tolerance(actual: dict, expected: dict, tolerance: dict) -> dict:
    """
    Compare expected metrics against actual metrics with tolerance.
    - If expected key exists in actual, compare directly.
    - If missing, mark as not comparable instead of raising KeyError.
    """
    report = {}
    for k, v in expected.items():
        e = float(v)
        if k not in actual:
            report[k] = {
                "actual": None,
                "expected": e,
                "tolerance": None,
                "pass": False,
                "reason": "missing_in_actual",
            }
            continue

        a = float(actual[k])
        t = float(tolerance.get(k, 0.0))
        ok = abs(a - e) <= t
        report[k] = {
            "actual": a,
            "expected": e,
            "tolerance": t,
            "pass": ok,
        }
    return report
