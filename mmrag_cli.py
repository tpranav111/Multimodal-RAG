from __future__ import annotations

import argparse
import json
import os

from mmrag_config import RagConfig
from mmrag_index import ingest_pdfs, reset_index
from mmrag_query import retrieve, answer_query


def main() -> None:
    parser = argparse.ArgumentParser(description="Local multimodal RAG (text + image).")
    parser.add_argument("--ingest", nargs="*", help="PDF paths to ingest")
    parser.add_argument("--query", type=str, help="Query string")
    parser.add_argument("--reset", action="store_true", help="Delete existing index")
    parser.add_argument("--llm", type=str, help="Override LLM model id")
    parser.add_argument("--no-cache", action="store_true", help="Disable retrieve/answer cache")
    parser.add_argument("--bm25", action="store_true", help="Enable BM25 + vector hybrid retrieval")
    parser.add_argument("--no-bm25", action="store_true", help="Disable BM25 hybrid retrieval")
    args = parser.parse_args()

    os.environ["HF_HUB_OFFLINE"] = "1"
    cfg = RagConfig()

    if args.llm:
        cfg.llm_model_id = args.llm
    if args.no_cache:
        cfg.enable_cache = False
    if args.bm25:
        cfg.enable_bm25 = True
    if args.no_bm25:
        cfg.enable_bm25 = False

    if args.reset:
        reset_index(cfg)

    if args.ingest:
        ingest_pdfs(args.ingest, cfg)
        print("Ingestion complete.")

    if args.query:
        context = retrieve(args.query, cfg)
        result = answer_query(context, cfg)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
