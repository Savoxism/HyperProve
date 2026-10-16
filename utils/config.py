from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    llm_api_key: str
    llm_base_url: str
    # Optional separate endpoint for embeddings (e.g. a second vLLM server);
    # falls back to the LLM endpoint when unset.
    embedding_base_url_override: str = ""
    # GPT-OSS-20B, the paper's backbone (served locally by scripts/serve_models.sh).
    llm_model: str = "openai/gpt-oss-20b"
    # gpt-oss reasoning effort (low | medium | high): offline entity/hyperedge extraction uses
    # index_reasoning_effort, every online step (decompose, rewrite, evaluate, answer, judge) uses
    # llm_reasoning_effort. High effort on extraction costs ~10x the tokens of medium.
    llm_reasoning_effort: str = "high"
    index_reasoning_effort: str = "medium"
    llm_temperature: float = 0.6
    # Cap on generated tokens (reasoning + answer) per call; at high effort gpt-oss occasionally
    # reasons for 30k+ tokens without stopping. A truncated call yields no answer.
    llm_max_tokens: int = 32768
    embedding_model: str = "BAAI/bge-m3"
    timeout: int = 3600

    @property
    def embedding_api_key(self) -> str:
        return self.llm_api_key

    @property
    def embedding_base_url(self) -> str:
        return self.embedding_base_url_override or self.llm_base_url

    def validate(self) -> None:
        missing = [
            name
            for name, value in (
                ("LLM_API_KEY", self.llm_api_key),
                ("LLM_BASE_URL", self.llm_base_url),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")


def load_settings() -> Settings:
    return Settings(
        llm_api_key=os.getenv("LLM_API_KEY", ""),
        llm_base_url=os.getenv("LLM_BASE_URL", ""),
        embedding_base_url_override=os.getenv("EMBEDDING_BASE_URL", ""),
        llm_reasoning_effort=os.getenv("LLM_REASONING_EFFORT", "high"),
        index_reasoning_effort=os.getenv("LLM_INDEX_REASONING_EFFORT", "medium"),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.6")),
        llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "32768")),
    )


settings = load_settings()


@dataclass
class HyperProveConfig:
    dataset_name: str
    dataset_path: str
    corpus_path: str
    output_root: str = "outputs_hyperprove"
    max_workers: int = 16
    llm_max_workers: int = 16
    judge_workers: int | None = None
    checkpoint_interval: int = 5
    rebuild: bool = False
    skip_index: bool = False
    index_only: bool = False
    retrieval_only: bool = False
    qa_only: bool = False
    skip_judge: bool = False
    limit_questions: int | None = None

    synonym_threshold: float = 0.8
    kc: int = 10
    kd: int = 10
    local_subgraph_size: int = 600
    top_hyperedges: int = 10
    qa_top_k: int = 10
    tau: float = 0.05
    theta: float = 0.4
    alpha: float = 0.5
    gamma: float = 0.85
    ppr_max_iter: int = 50

    @property
    def run_dir(self) -> str:
        model = settings.llm_model.replace("/", "_").replace(":", "_")
        embed = settings.embedding_model.replace("/", "_").replace(":", "_")
        return os.path.join(self.output_root, self.dataset_name, f"{model}_{embed}")

    @property
    def artifacts_dir(self) -> str:
        return os.path.join(self.run_dir, "artifacts")

    @property
    def logs_dir(self) -> str:
        return "logs"

    @property
    def retrieval_results_path(self) -> str:
        return os.path.join(self.run_dir, "retrieval_results.json")

    @property
    def index_manifest_path(self) -> str:
        return os.path.join(self.artifacts_dir, "manifest.json")
