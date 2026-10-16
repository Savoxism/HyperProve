"""Stable-key checkpoint utilities shared by offline and online stages."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Hashable, TypeVar

from tqdm.auto import tqdm


T = TypeVar("T")


def rows_from_payload(payload: Any, container_key: str | None = None) -> list[dict[str, Any]]:
    rows = payload.get(container_key) if container_key and isinstance(payload, dict) else payload
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def merge_records(
    existing: list[dict[str, Any]],
    current: list[Any],
    key: Callable[[dict[str, Any]], Hashable],
) -> list[dict[str, Any]]:
    merged: dict[Hashable, dict[str, Any]] = {}
    order: list[Hashable] = []
    for row in [*existing, *(item for item in current if isinstance(item, dict))]:
        row_key = key(row)
        if row_key not in merged:
            order.append(row_key)
        merged[row_key] = row
    return [merged[row_key] for row_key in order]


def run_checkpointed_parallel(
    *,
    results: list[T | None],
    desc: str,
    max_workers: int,
    checkpoint_interval: int,
    submit_one: Callable[[int], T],
    write_checkpoint: Callable[[list[T | None]], None],
) -> list[T]:
    total = len(results)
    completed = sum(item is not None for item in results)
    pending = [idx for idx, item in enumerate(results) if item is None]
    interval = max(1, int(checkpoint_interval))
    with tqdm(total=total, initial=completed, desc=desc, dynamic_ncols=True) as progress:
        with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
            future_to_index = {executor.submit(submit_one, idx): idx for idx in pending}
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                results[index] = future.result()
                completed += 1
                progress.update(1)
                if completed % interval == 0:
                    write_checkpoint(results)
    write_checkpoint(results)
    if any(item is None for item in results):
        raise RuntimeError(f"Checkpointed stage {desc!r} completed with missing records.")
    return [item for item in results if item is not None]
