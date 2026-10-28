"""Append-only generation log (JSON lines). Never stores script text."""
import json
from datetime import datetime
from pathlib import Path


def log_event(path: str | Path, **fields) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"time": datetime.now().isoformat(timespec="seconds"), **fields}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
