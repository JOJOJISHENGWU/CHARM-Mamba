from __future__ import annotations

import json
from pathlib import Path


def ensure_exists(path: str) -> None:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")


def main() -> None:
    required = [
        "REPRODUCIBILITY.md",
        "BASELINE_FAIRNESS.md",
        "EFFICIENCY_REPORT.md",
        "results/expected_metrics.json",
        "checkpoints/MANIFEST.json",
        "checkpoints/SHA256SUMS.txt",
        "logs/README.md",
    ]
    for p in required:
        ensure_exists(p)

    manifest = json.loads(Path("checkpoints/MANIFEST.json").read_text(encoding="utf-8"))
    if "artifacts" not in manifest or not isinstance(manifest["artifacts"], list):
        raise ValueError("Invalid manifest format: key 'artifacts' is required.")

    print("Artifact structure verification passed.")


if __name__ == "__main__":
    main()
