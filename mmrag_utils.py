from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Iterable, List, Dict, Tuple

import fitz  # PyMuPDF


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start = max(end - chunk_overlap, end)
    return chunks


def extract_pdf_text_and_images(
    pdf_path: Path,
    images_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> Tuple[List[Dict], List[Dict]]:
    doc = fitz.open(str(pdf_path))
    text_items: List[Dict] = []
    image_items: List[Dict] = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_text = page.get_text("text") or ""
        chunks = chunk_text(page_text, chunk_size, chunk_overlap)

        for chunk_idx, chunk in enumerate(chunks):
            text_items.append(
                {
                    "id": f"{pdf_path.name}-p{page_idx}-c{chunk_idx}-{uuid.uuid4()}",
                    "type": "text",
                    "page": page_idx,
                    "content": chunk,
                    "source": str(pdf_path),
                }
            )

        images = page.get_images(full=True)
        for img_idx, img in enumerate(images):
            xref = img[0]
            base = doc.extract_image(xref)
            img_bytes = base["image"]
            img_ext = base.get("ext", "png")
            img_name = f"{pdf_path.stem}_p{page_idx}_i{img_idx}.{img_ext}"
            img_path = images_dir / img_name
            img_path.write_bytes(img_bytes)

            image_items.append(
                {
                    "id": f"{pdf_path.name}-p{page_idx}-i{img_idx}-{uuid.uuid4()}",
                    "type": "image",
                    "page": page_idx,
                    "content": f"Image from page {page_idx} of {pdf_path.name}",
                    "image_path": str(img_path),
                    "source": str(pdf_path),
                }
            )

    doc.close()
    return text_items, image_items
