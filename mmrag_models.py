from __future__ import annotations

from typing import List, Iterable

import torch
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    CLIPModel,
    CLIPProcessor,
    pipeline,
)


def load_text_embedder(model_id: str, device: str) -> SentenceTransformer:
    return SentenceTransformer(
        model_id,
        device=device,
        cache_folder=None,
        local_files_only=True,
    )


def embed_texts(embedder: SentenceTransformer, texts: List[str]) -> List[List[float]]:
    return embedder.encode(texts, normalize_embeddings=True).tolist()


def load_clip(model_id: str, device: str):
    clip_model = CLIPModel.from_pretrained(model_id, local_files_only=True)
    clip_processor = CLIPProcessor.from_pretrained(model_id, local_files_only=True)
    if device == "cuda":
        clip_model = clip_model.to(device)
    return clip_model, clip_processor


def embed_clip_texts(clip_model, clip_processor, texts: List[str], device: str) -> List[List[float]]:
    inputs = clip_processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    if device == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        feats = clip_model.get_text_features(**inputs)
    feats = torch.nn.functional.normalize(feats, p=2, dim=1)
    return feats.cpu().tolist()


def embed_clip_images(clip_model, clip_processor, images: List, device: str) -> List[List[float]]:
    inputs = clip_processor(images=images, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        feats = clip_model.get_image_features(**inputs)
    feats = torch.nn.functional.normalize(feats, p=2, dim=1)
    return feats.cpu().tolist()


def load_llm(model_id: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        local_files_only=True,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if device == "cuda" else -1,
    )
