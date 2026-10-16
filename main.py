from __future__ import annotations

import argparse
import json
import logging

from utils.config import HyperProveConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HyperProve answer-guided hypergraph expansion pipeline")
    parser.add_argument("--dataset-name", required=True, help="Logical dataset name used for output directory naming.")
    parser.add_argument("--dataset-path", required=True, help="Path to benchmark question JSON.")
    parser.add_argument("--corpus-path", required=True, help="Path to benchmark corpus JSON.")
    parser.add_argument("--output-root", default="outputs_hyperprove")
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--llm-max-workers", type=int, default=16)
    parser.add_argument("--judge-workers", type=int, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    parser.add_argument("--rebuild", action="store_true", help="Rebuild all offline HyperProve artifacts.")
    parser.add_argument("--skip-index", action="store_true", help="Reuse existing HyperProve index artifacts.")
    parser.add_argument("--index-only", action="store_true", help="Build the offline hyperedge index and stop.")
    parser.add_argument("--retrieval-only", action="store_true", help="Run retrieval and stop before QA/evaluation.")
    parser.add_argument("--qa-only", action="store_true", help="Reuse retrieval_results.json and rerun QA/evaluation.")
    parser.add_argument("--skip-judge", action="store_true", help="Skip LLM-as-a-judge evaluation.")
    parser.add_argument("--limit-questions", type=int, default=None, help="Optional cap for smoke tests.")

    parser.add_argument("--synonym-threshold", type=float, default=0.8)
    parser.add_argument("--kc", type=int, default=10, help="Carried-frontier seed count per hop.")
    parser.add_argument("--kd", type=int, default=10, help="Fresh dense seed count per hop.")
    parser.add_argument("--local-subgraph-size", type=int, default=600)
    parser.add_argument("--top-hyperedges", type=int, default=10)
    parser.add_argument("--qa-top-k", type=int, default=10)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--theta", type=float, default=0.4)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.85)
    parser.add_argument("--ppr-max-iter", type=int, default=50)
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    for noisy_logger in ("httpx", "httpcore", "openai", "openai._base_client"):
        logging.getLogger(noisy_logger).setLevel(logging.ERROR)
    args = build_parser().parse_args()
    config = HyperProveConfig(
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        corpus_path=args.corpus_path,
        output_root=args.output_root,
        max_workers=args.max_workers,
        llm_max_workers=args.llm_max_workers,
        judge_workers=args.judge_workers,
        checkpoint_interval=args.checkpoint_interval,
        rebuild=args.rebuild,
        skip_index=args.skip_index,
        index_only=args.index_only,
        retrieval_only=args.retrieval_only,
        qa_only=args.qa_only,
        skip_judge=args.skip_judge,
        limit_questions=args.limit_questions,
        synonym_threshold=args.synonym_threshold,
        kc=args.kc,
        kd=args.kd,
        local_subgraph_size=args.local_subgraph_size,
        top_hyperedges=args.top_hyperedges,
        qa_top_k=args.qa_top_k,
        tau=args.tau,
        theta=args.theta,
        alpha=args.alpha,
        gamma=args.gamma,
        ppr_max_iter=args.ppr_max_iter,
    )
    from src.pipeline import run_pipeline

    summary = run_pipeline(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
