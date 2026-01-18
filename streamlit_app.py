from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from mmrag_config import RagConfig
from mmrag_index import ingest_pdfs, reset_index
from mmrag_query import retrieve, answer_query
from mmrag_utils import ensure_dirs


st.set_page_config(page_title="Local Multimodal RAG", layout="wide")


def list_pdfs(pdf_dir: Path) -> list[str]:
    if not pdf_dir.exists():
        return []
    return [str(p) for p in sorted(pdf_dir.glob("*.pdf"))]


def save_uploaded_pdfs(upload_dir: Path, uploads) -> list[str]:
    saved = []
    for up in uploads:
        out_path = upload_dir / up.name
        out_path.write_bytes(up.getbuffer())
        saved.append(str(out_path))
    return saved


cfg = RagConfig()
ensure_dirs(cfg.data_dir, cfg.images_dir, cfg.index_dir, cfg.cache_dir)

st.title("Local Multimodal RAG")
st.caption("Index PDFs and query with text + image retrieval using local models.")
image_width = 320

with st.sidebar:
    st.header("Index Controls")
    st.write(f"Device: `{cfg.device}`")
    st.write(f"Text embedder: `{cfg.text_embed_model_id}`")
    st.write(f"CLIP: `{cfg.clip_model_id}`")
    st.write(f"LLM: `{cfg.llm_model_id}`")
    cfg.enable_cache = st.toggle("Enable cache", value=cfg.enable_cache)
    cfg.enable_bm25 = st.toggle("Enable BM25 hybrid retrieval", value=cfg.enable_bm25)
    cfg.hybrid_weight = st.slider(
        "Hybrid weight (vector vs BM25)",
        min_value=0.0,
        max_value=1.0,
        value=cfg.hybrid_weight,
        step=0.05,
    )

    if st.button("Reset Vector Index", use_container_width=True):
        reset_index(cfg)
        st.success("Index reset.")

    st.subheader("Upload PDFs")
    uploads = st.file_uploader("Add PDFs", type=["pdf"], accept_multiple_files=True)
    if uploads and st.button("Save Uploads", use_container_width=True):
        upload_dir = cfg.data_dir / "input" / "pdfs"
        ensure_dirs(upload_dir)
        saved = save_uploaded_pdfs(upload_dir, uploads)
        st.success(f"Saved {len(saved)} PDF(s).")

    st.subheader("Ingest PDFs")
    pdf_dir = cfg.data_dir / "input" / "pdfs"
    existing_pdfs = list_pdfs(pdf_dir)
    st.write(f"Found {len(existing_pdfs)} PDF(s) in `{pdf_dir}`")
    if st.button("Ingest All PDFs", use_container_width=True):
        ingest_pdfs(existing_pdfs, cfg)
        st.success("Ingestion complete.")

st.subheader("Query")
query = st.text_input("Ask a question", value="Explain the key ideas of attention in neural networks.")
run = st.button("Run Query", type="primary")

if run and query.strip():
    with st.spinner("Retrieving..."):
        context = retrieve(query, cfg)
    with st.spinner("Generating answer..."):
        result = answer_query(context, cfg)

    st.markdown("### Answer")
    st.write(result.get("answer", ""))

    st.markdown("### Selected Images")
    for img_path in result.get("image_paths", []):
        st.image(img_path, caption=img_path, width=image_width)

    with st.expander("Retrieved Text Context", expanded=False):
        st.code("\n\n".join(context.get("text_context", [])))

    with st.expander("Retrieved Image Candidates", expanded=False):
        for img in context.get("images", []):
            st.image(img["image_path"], caption=f"page {img['page']}", width=image_width)

    with st.expander("Raw JSON", expanded=False):
        st.code(json.dumps({"context": context, "result": result}, indent=2))
