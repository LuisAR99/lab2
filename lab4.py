# app.py
import streamlit as st
from openai import OpenAI
import tiktoken
import uuid
from pathlib import Path
import glob
import os
import sys

# ---------------------------------------------------------
# ✅ FIX for ChromaDB on Streamlit Cloud (use modern SQLite)
# ---------------------------------------------------------
# Matches your slide: import pysqlite3 and swap stdlib sqlite3
__import__("pysqlite3")  # ensure wheel is present (pysqlite3-binary)
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
# (dbapi2 is optional; most libs import sqlite3 directly)

# Now safe to import chroma
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# Use PyPDF2 as in your slide
from PyPDF2 import PdfReader

# =========================
# Page & App Settings
# =========================
st.set_page_config(page_title="Lab 4 — Vector Similarity", page_icon="💬")
st.title("💬 Lab 4 — Vector Similarity")

# Folder where your local PDFs live (commit them to your repo)
PDF_DIR = "docs"                  # e.g., ./docs
CHROMA_PATH = "./ChromaDB_for_lab"  # matches your slide

# =========================
# Secrets & OpenAI client
# =========================
if "openai_client" not in st.session_state:
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except KeyError:
        st.error("Missing `OPENAI_API_KEY` in Streamlit secrets.")
        st.stop()
    st.session_state.openai_client = OpenAI(api_key=api_key, timeout=60, max_retries=2)

client = st.session_state.openai_client

# =========================
# Session State
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_handled_prompt" not in st.session_state:
    st.session_state.last_handled_prompt = None
# Will hold the Chroma collection
# st.session_state.Lab4_vectorDB

# =========================
# UI: Model & Clear
# =========================
col1, col2 = st.columns(2)
with col1:
    model_name = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"], index=0)
with col2:
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.last_handled_prompt = None
        # st.session_state.pop("Lab4_vectorDB", None)  # uncomment for full reset
        st.rerun()

# =========================
# Helpers: token trimming
# =========================
def _get_encoding(model: str):
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")

def trim_messages_to_tokens(messages, max_tokens=200, model="gpt-4o-mini"):
    enc = _get_encoding(model)
    result, total = [], 0
    for m in reversed(messages):
        t = len(enc.encode(m.get("content", "")))
        if total + t > max_tokens:
            break
        result.append(m)
        total += t
    return list(reversed(result))

SYSTEM_PROMPT = (
    "You are a friendly chatbot explaining things so a 10-year-old can understand. "
    "Use short, clear sentences and simple words. "
    "Whenever you answer, finish with exactly this question on a new line: DO YOU WANT MORE INFO?"
    "If the user says yes then provide more information and re-ask DO YOU WANT MORE INFO"
    "If the user says no ask the user what question can the bot help with"
    "If you are unsure re-ask DO YOU WANT MORE INFO"
)

# =========================
# PDF → Text, Chunking, ChromaDB (PersistentClient)
# =========================
def _pdf_to_text_from_path(pdf_path: Path) -> str:
    with pdf_path.open("rb") as f:
        reader = PdfReader(f)
        pages = []
        for p in reader.pages:
            try:
                pages.append(p.extract_text() or "")
            except Exception:
                pages.append("")
        return "\n".join(pages).strip()

def _chunk_text(text: str, chunk_chars: int = 2000, overlap: int = 200):
    text = text.replace("\r", "")
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + chunk_chars, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(end - overlap, 0)
    return chunks

def _find_local_pdfs(root_dir: str):
    pattern = str(Path(root_dir) / "**" / "*.pdf")
    return [Path(p) for p in glob.glob(pattern, recursive=True)]

def get_or_create_lab4_vdb_from_local(pdf_dir: str):
    """
    Build (or return) Chroma 'Lab4Collection' using PersistentClient at CHROMA_PATH.
    Only builds once per app run; collection handle stored in st.session_state.Lab4_vectorDB.
    """
    if "Lab4_vectorDB" in st.session_state and st.session_state.Lab4_vectorDB is not None:
        return st.session_state.Lab4_vectorDB

    # Ensure path exists
    Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)

    # ✅ Your slide's approach: PersistentClient with on-disk store
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    embedder = OpenAIEmbeddingFunction(
        api_key=client.api_key,          # reuse the same key
        model_name="text-embedding-3-small"
    )

    collection = chroma_client.get_or_create_collection(
        name="Lab4Collection",
        embedding_function=embedder,
        metadata={"hnsw:space": "cosine"}
    )

    # Ingest local PDFs (skip ones already present)
    pdf_paths = _find_local_pdfs(pdf_dir)
    if not pdf_paths:
        st.info(f"No PDFs found under '{pdf_dir}'.")
        st.session_state.Lab4_vectorDB = collection
        return collection

    existing_ids = set()
    try:
        existing = collection.get()
        existing_ids = set(existing.get("ids", []))
    except Exception:
        pass

    for i, p in enumerate(pdf_paths):
        doc_key = f"{i:04d}_{p.name}"
        if any(eid.startswith(doc_key) for eid in existing_ids):
            continue

        try:
            raw_text = _pdf_to_text_from_path(p)
        except Exception as e:
            st.warning(f"Could not read {p.name}: {e}")
            continue
        if not raw_text:
            continue

        chunks = _chunk_text(raw_text, chunk_chars=2000, overlap=200)
        ids, docs, metas = [], [], []
        for idx, chunk in enumerate(chunks):
            uid = f"{doc_key}_chunk_{idx}_{uuid.uuid4().hex[:8]}"
            ids.append(uid)
            docs.append(chunk)
            metas.append({
                "doc_key": doc_key,
                "source_path": str(p),
                "source_name": p.name,
                "chunk_index": idx,
                "total_chunks": len(chunks),
            })
        if ids:
            collection.add(ids=ids, documents=docs, metadatas=metas)

    st.session_state.Lab4_vectorDB = collection
    return collection

# =========================
# Build Vector DB once
# =========================
with st.sidebar:
    st.subheader("📚 Local Docs (SQLite via pysqlite3)")
    st.caption(f"Loading PDFs from: `{PDF_DIR}`")
    if "Lab4_vectorDB" not in st.session_state:
        with st.spinner("Building Lab4Collection from local PDFs (one-time; persisted)…"):
            vdb = get_or_create_lab4_vdb_from_local(PDF_DIR)
            st.success(f"Vector DB ready. Chunks: ~{vdb.count()}")

    # Optional quick query UI
    vdb = st.session_state.get("Lab4_vectorDB")
    if vdb:
        st.divider()
        q = st.text_input("Quick search your docs")
        if q:
            res = vdb.query(query_texts=[q], n_results=5)
            st.write(res)

# =========================
# Chat UI
# =========================
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

prompt = st.chat_input("Ask me anything…")
if prompt and prompt != st.session_state.last_handled_prompt:
    st.session_state.last_handled_prompt = prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    limited_history = trim_messages_to_tokens(st.session_state.messages, max_tokens=200, model=model_name)
    model_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + limited_history

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=model_name,
            messages=model_messages,
            stream=True,
            timeout=60,
        )
        assistant_text = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": assistant_text})
    st.rerun()

