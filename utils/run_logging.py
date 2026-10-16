"""Thread-safe, run-scoped logging for the three HyperProve stages."""

from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

from utils.io import ensure_dir


Stage = Literal["indexing", "inference", "evaluate"]
_STAGES: tuple[Stage, ...] = ("indexing", "inference", "evaluate")
_LOCK = threading.RLock()
_PATHS: dict[Stage, str] = {}


def _slug(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", str(text).strip()).strip("-._")
    return value or "run"


def configure_run_logs(dataset_name: str, logs_dir: str = "logs") -> dict[Stage, str]:
    """Create exactly one log file per stage and return their paths."""
    timestamp = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).strftime("%Y%m%dT%H%M%S%z")
    ensure_dir(logs_dir)
    prefix = _slug(dataset_name)
    with _LOCK:
        _PATHS.clear()
        for stage in _STAGES:
            path = os.path.join(logs_dir, f"{prefix}_{stage}_{timestamp}.log")
            Path(path).write_text(
                f"HyperProve stage={stage} dataset={dataset_name} run_timestamp={timestamp}\n",
                encoding="utf-8",
            )
            _PATHS[stage] = path
    return dict(_PATHS)


def get_log_paths() -> dict[Stage, str]:
    return dict(_PATHS)


def log_event(stage: Stage, event: str, *, level: str = "INFO", **payload: Any) -> None:
    path = _PATHS.get(stage)
    if not path:
        return
    now = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat(timespec="seconds")
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) if payload else "{}"
    line = f"{now} | {level.upper()} | {event} | {serialized}\n"
    with _LOCK:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line)


def stage_for_metadata(metadata: dict[str, Any] | None) -> Stage:
    step = str((metadata or {}).get("step", "")).lower()
    if step in {"judge", "evaluation"}:
        return "evaluate"
    if step in {"entity_extraction", "hyperedge_extraction", "indexing", "embedding_index"}:
        return "indexing"
    return "inference"
