from __future__ import annotations

import logging
import random
import threading
import time
from typing import Any

from utils.api import is_retryable_api_error
from utils.config import settings
from utils.run_logging import log_event, stage_for_metadata

from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMClient:
    # One OpenAI client (and connection pool) shared by every instance and worker
    # thread; a client per call leaks sockets until the process hits its fd limit.
    _client: Any | None = None
    _client_lock = threading.Lock()

    @property
    def client(self) -> Any:
        with self._client_lock:
            if type(self)._client is not None:
                return type(self)._client
            if OpenAI is None:
                raise RuntimeError("The openai package is required for LLM calls.")
            settings.validate()
            type(self)._client = OpenAI(
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
                timeout=settings.timeout,
            )
            return type(self)._client

    def call(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_retries: int = 10,
        artifact_metadata: dict[str, Any] | None = None,
        return_metadata: bool = False,
        reasoning_effort: str | None = None,
    ) -> str | dict[str, Any]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=settings.llm_model,
                    messages=messages,
                    temperature=settings.llm_temperature,
                    reasoning_effort=reasoning_effort or settings.llm_reasoning_effort,
                    max_tokens=settings.llm_max_tokens,
                )
                choice = response.choices[0]
                content = choice.message.content or ""
                reasoning = getattr(choice.message, "reasoning_content", None)
                if return_metadata:
                    return {"content": content.strip(), "reasoning_content": reasoning}
                return content.strip()
            except Exception as exc:
                last_error = exc
                stage = stage_for_metadata(artifact_metadata)
                log_event(
                    stage,
                    "llm_call_retry",
                    level="WARNING",
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    error=repr(exc),
                    metadata=artifact_metadata or {},
                )
                if not is_retryable_api_error(exc):
                    break
                if attempt >= max_retries - 1:
                    break
                logger.warning("Retrying LLM call %s/%s after error: %s", attempt + 1, max_retries, exc)
                time.sleep(min((2.0 ** attempt) + random.uniform(0.0, 0.5), 30.0))
        log_event(
            stage_for_metadata(artifact_metadata),
            "llm_call_failed",
            level="ERROR",
            error=repr(last_error),
            metadata=artifact_metadata or {},
            system_prompt=system_prompt or "",
            prompt=prompt,
        )
        raise RuntimeError(f"LLM call failed after {max_retries} retries.") from last_error
