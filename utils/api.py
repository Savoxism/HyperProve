"""Shared retry classification for OpenAI-compatible backends."""

from __future__ import annotations


def is_retryable_api_error(error: Exception) -> bool:
    status = getattr(error, "status_code", None)
    if status is None:
        return True
    return int(status) in {408, 409, 429} or int(status) >= 500
