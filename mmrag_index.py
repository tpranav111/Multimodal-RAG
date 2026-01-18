from __future__ import annotations

from pathlib import Path
from typing import List

import chromadb
from PIL import Image

from mmrag_config import RagConfig
from mmrag_models import load_text_embedder, embed_texts, load_clip, embed_clip_images
from mmrag_utils import ensure_dirs, extract_pdf_text_and_images


def reset_index(cfg: RagConfig) -> None:
    client = chromadb.PersistentClient(path=str(cfg.index_dir))
    try:
        client.delete_collection("text_chunks")
    except Exception:
        pass
    try:
        client.delete_collection("image_items")
    except Exception:
        pass


def ingest_pdfs(pdf_paths: List[str], cfg: RagConfig) -> None:
    ensure_dirs(cfg.data_dir, cfg.images_dir, cfg.index_dir)

    text_embedder = load_text_embedder(cfg.text_embed_model_id, cfg.device)
    clip_model, clip_processor = load_clip(cfg.clip_model_id, cfg.device)

    client = chromadb.PersistentClient(path=str(cfg.index_dir))
    text_col = client.get_or_create_collection("text_chunks")
    image_col = client.get_or_create_collection("image_items")

    for pdf in pdf_paths:
        pdf_path = Path(pdf)
        text_items, image_items = extract_pdf_text_and_images(
            pdf_path=pdf_path,
            images_dir=cfg.images_dir,
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
        )

        if text_items:
            texts = [t["content"] for t in text_items]
            text_embs = embed_texts(text_embedder, texts)
            text_col.add(
                ids=[t["id"] for t in text_items],
                documents=texts,
                embeddings=text_embs,
                metadatas=[
                    {
                        "page": t["page"],
                        "source": t["source"],
                        "type": t["type"],
                    }
                    for t in text_items
                ],
            )

        if image_items:
            images = []
            for item in image_items:
                img = Image.open(item["image_path"]).convert("RGB")
                images.append(img)
            image_embs = embed_clip_images(clip_model, clip_processor, images, cfg.device)
            for img in images:
                try:
                    img.close()
                except Exception:
                    pass
            image_col.add(
                ids=[i["id"] for i in image_items],
                documents=[i["content"] for i in image_items],
                embeddings=image_embs,
                metadatas=[
                    {
                        "page": i["page"],
                        "source": i["source"],
                        "type": i["type"],
                        "image_path": i["image_path"],
                    }
                    for i in image_items
                ],
            )
