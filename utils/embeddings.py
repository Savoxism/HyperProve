from __future__ import annotations

import logging
import random
import threading
import time
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from utils.api import is_retryable_api_error
from utils.config import settings
from utils.run_logging import Stage, log_event

from openai import OpenAI


logger = logging.getLogger(__name__)


class EmbeddingGenerator:
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
                raise RuntimeError("The openai package is required for embedding calls.")
            settings.validate()
            type(self)._client = OpenAI(
                api_key=settings.embedding_api_key,
                base_url=settings.embedding_base_url,
            )
            return type(self)._client

    def batch_encode(
        self,
        texts: list[str],
        batch_size: int = 128,
        show_progress: bool = True,
        stage: Stage = "inference",
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        outputs: list[list[float]] = []
        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
        iterator = tqdm(batches, desc="Embedding") if show_progress else batches
        for batch in iterator:
            outputs.extend(self._embed_batch(batch, stage=stage))
        arr = np.asarray(outputs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(norms, 1e-12)

    def _embed_batch(
        self,
        texts: list[str],
        max_retries: int = 10,
        stage: Stage = "inference",
    ) -> list[list[float]]:
        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    model=settings.embedding_model,
                    input=texts,
                )
                return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
            except Exception as exc:
                last_error = exc
                log_event(
                    stage,
                    "embedding_call_retry",
                    level="WARNING",
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    batch_size=len(texts),
                    error=repr(exc),
                )
                if not is_retryable_api_error(exc):
                    break
                if attempt >= max_retries - 1:
                    break
                logger.warning("Retrying embedding call %s/%s after error: %s", attempt + 1, max_retries, exc)
                time.sleep(min((2.0 ** attempt) + random.uniform(0.0, 0.5), 30.0))
        log_event(stage, "embedding_call_failed", level="ERROR", error=repr(last_error), batch_size=len(texts))
        raise RuntimeError(f"Embedding call failed after {max_retries} retries.") from last_error
