from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class RagConfig:
    # Local model ids (must exist in HF cache)
    text_embed_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    clip_model_id: str = "openai/clip-vit-base-patch32"
    llm_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"

    # Storage paths
    project_root: Path = Path(__file__).resolve().parent
    data_dir: Path = project_root / "data"
    images_dir: Path = data_dir / "images"
    index_dir: Path = project_root / "index"
    cache_dir: Path = project_root / "cache"

    # Caching
    enable_cache: bool = True

    # Chunking
    chunk_size: int = 900
    chunk_overlap: int = 150

    # Retrieval
    top_k_text: int = 6
    top_k_images: int = 4
    enable_bm25: bool = True
    bm25_top_k: int = 20
    hybrid_weight: float = 0.6

    # Generation
    max_new_tokens: int = 400
    temperature: float = 0.2

    # Runtime
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
