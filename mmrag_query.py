from __future__ import annotations

import json
import re
from typing import Dict, List

import chromadb

from mmrag_config import RagConfig
from mmrag_cache import cache_get, cache_set
from mmrag_models import (
    load_text_embedder,
    embed_texts,
    load_clip,
    embed_clip_texts,
    load_llm,
)


def retrieve(query: str, cfg: RagConfig) -> Dict:
    cache_payload = {
        "query": query,
        "text_model": cfg.text_embed_model_id,
        "clip_model": cfg.clip_model_id,
        "top_k_text": cfg.top_k_text,
        "top_k_images": cfg.top_k_images,
        "enable_bm25": cfg.enable_bm25,
        "bm25_top_k": cfg.bm25_top_k,
        "hybrid_weight": cfg.hybrid_weight,
    }
    if cfg.enable_cache:
        cached = cache_get(cfg.cache_dir, "retrieve", cache_payload)
        if cached is not None:
            return cached

    client = chromadb.PersistentClient(path=str(cfg.index_dir))
    text_col = client.get_or_create_collection("text_chunks")
    image_col = client.get_or_create_collection("image_items")

    text_embedder = load_text_embedder(cfg.text_embed_model_id, cfg.device)
    clip_model, clip_processor = load_clip(cfg.clip_model_id, cfg.device)

    text_emb = embed_texts(text_embedder, [query])[0]
    clip_text_emb = embed_clip_texts(clip_model, clip_processor, [query], cfg.device)[0]

    text_res = text_col.query(
        query_embeddings=[text_emb],
        n_results=cfg.top_k_text,
        include=["documents", "metadatas", "distances"],
    )
    image_res = image_col.query(query_embeddings=[clip_text_emb], n_results=cfg.top_k_images)

    text_docs = text_res.get("documents", [[]])[0]
    text_metas = text_res.get("metadatas", [[]])[0]
    text_ids = text_res.get("ids", [[]])[0]
    text_dists = text_res.get("distances", [[]])[0]

    if cfg.enable_bm25:
        try:
            from rank_bm25 import BM25Okapi
        except Exception as exc:
            raise RuntimeError(
                "BM25 enabled but rank_bm25 is not installed. "
                "Install with: python -m pip install rank_bm25"
            ) from exc

        def tokenize(txt: str) -> List[str]:
            return re.findall(r"[A-Za-z0-9]+", txt.lower())

        all_docs = text_col.get(include=["documents"])
        corpus_docs = all_docs.get("documents", [])
        corpus_ids = all_docs.get("ids", [])
        tokenized = [tokenize(doc) for doc in corpus_docs]
        bm25 = BM25Okapi(tokenized)
        scores = bm25.get_scores(tokenize(query))
        scored = sorted(zip(corpus_ids, corpus_docs, scores), key=lambda x: x[2], reverse=True)
        bm25_top = scored[: cfg.bm25_top_k]

        vec_scores = {}
        for _id, dist in zip(text_ids, text_dists):
            vec_scores[_id] = max(0.0, 1.0 - float(dist))

        bm25_scores = {}
        max_bm25 = max([s for _, _, s in bm25_top], default=0.0)
        for _id, doc, score in bm25_top:
            bm25_scores[_id] = 0.0 if max_bm25 == 0 else float(score) / max_bm25

        combined = {}
        for _id in set(list(vec_scores.keys()) + list(bm25_scores.keys())):
            v = vec_scores.get(_id, 0.0)
            b = bm25_scores.get(_id, 0.0)
            combined[_id] = (cfg.hybrid_weight * v) + ((1.0 - cfg.hybrid_weight) * b)

        id_to_doc = dict(zip(text_ids, text_docs))
        id_to_meta = dict(zip(text_ids, text_metas))
        for _id, doc, _score in bm25_top:
            if _id not in id_to_doc:
                id_to_doc[_id] = doc

        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[: cfg.top_k_text]
        text_docs = [id_to_doc[_id] for _id, _ in ranked]
        text_metas = [id_to_meta.get(_id, {}) for _id, _ in ranked]

    image_docs = image_res.get("documents", [[]])[0]
    image_meta = image_res.get("metadatas", [[]])[0]

    images = []
    for meta, desc in zip(image_meta, image_docs):
        images.append(
            {
                "image_path": meta.get("image_path", ""),
                "page": meta.get("page"),
                "desc": desc,
            }
        )

    result = {
        "query": query,
        "text_context": text_docs,
        "images": images,
    }
    if cfg.enable_cache:
        cache_set(cfg.cache_dir, "retrieve", cache_payload, result)
    return result


def answer_query(context: Dict, cfg: RagConfig) -> Dict:
    cache_payload = {
        "query": context.get("query", ""),
        "text_context": context.get("text_context", []),
        "images": context.get("images", []),
        "llm_model": cfg.llm_model_id,
        "temperature": cfg.temperature,
        "max_new_tokens": cfg.max_new_tokens,
    }
    if cfg.enable_cache:
        cached = cache_get(cfg.cache_dir, "answer", cache_payload)
        if cached is not None:
            return cached

    llm = load_llm(cfg.llm_model_id, cfg.device)

    image_lines = []
    for i, img in enumerate(context["images"]):
        image_lines.append(
            f"[{i}] page={img['page']} path={img['image_path']} desc={img['desc']}"
        )

    prompt = (
        "You are a helpful assistant. Answer the question using the context.\n"
        "Then choose the most relevant image indices (or [] if none).\n\n"
        f"Question: {context['query']}\n\n"
        "Text context:\n"
        + "\n".join(context["text_context"])
        + "\n\n"
        "Image candidates:\n"
        + "\n".join(image_lines)
        + "\n\n"
        "Return JSON with keys: answer (string), image_indices (list of ints).\n"
    )

    output = llm(
        prompt,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        do_sample=cfg.temperature > 0,
    )[0]["generated_text"]

    json_start = output.find("{")
    json_end = output.rfind("}")
    if json_start == -1 or json_end == -1:
        return {"answer": output.strip(), "image_paths": []}

    try:
        payload = json.loads(output[json_start : json_end + 1])
    except Exception:
        return {"answer": output.strip(), "image_paths": []}

    image_paths = []
    for idx in payload.get("image_indices", []):
        if isinstance(idx, int) and 0 <= idx < len(context["images"]):
            image_paths.append(context["images"][idx]["image_path"])

    result = {
        "answer": payload.get("answer", ""),
        "image_paths": image_paths,
    }
    if cfg.enable_cache:
        cache_set(cfg.cache_dir, "answer", cache_payload, result)
    return result
