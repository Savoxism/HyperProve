from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from typing import Any, Iterator


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


@contextmanager
def atomic_output_path(path: str, suffix: str) -> Iterator[str]:
    directory = os.path.dirname(path)
    ensure_dir(directory)
    fd, tmp_path = tempfile.mkstemp(dir=directory or None, prefix=".tmp_", suffix=suffix)
    os.close(fd)
    try:
        yield tmp_path
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_json_if_exists(path: str) -> Any | None:
    if not os.path.exists(path):
        return None
    return read_json(path)


def write_json(path: str, payload: Any) -> None:
    with atomic_output_path(path, ".json") as tmp_path:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_text(path: str, text: str) -> None:
    with atomic_output_path(path, ".txt") as tmp_path:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            handle.write(text)

