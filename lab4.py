# app.py
import streamlit as st
from openai import OpenAI
import tiktoken
import uuid
from pathlib import Path
import glob

# --- Vectors: Chroma + PDF imports ---
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from pypdf import PdfReader

# =========================
# Page & App Settings
# =========================
st.set_page_config(page_title="Lab 4 — Vector Similarity", page_icon="💬")
st.title("💬 Lab 4 — Vector Similarity")

# Folder where your local PDFs live (relative to this app.py)
PDF_DIR = "docs"   # e.g., put PDFs in ./docs ; can be nested; change as needed

# =========================
# Secrets & OpenAI client
# =========================
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("Missing `OPENAI_API_KEY` in Streamlit secrets.")
    st.stop()

client = OpenAI(api_key=api_key, timeout=60, max_retries=2)

# =========================
# Session State
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_handled_prompt" not in st.session_state:
    st.session_state.last_handled_prompt = None
# Vector DB handle will be stored here once created:
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
        # If you also want to clear the vector DB for this run, uncomment:
        # st.session_state.pop("Lab4_vectorDB", None)
        st.rerun()

# =========================
# Helpers: token trimming
# =========================
def _get_encoding(model: str):
    """Safe tiktoken encoding getter with fallback."""
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")

def trim_messages_to_tokens(messages, max_tokens=200, model="gpt-4o-mini"):
    enc = _get_encoding(model)
    result = []
    total = 0
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
# PDF → Text, Chunking, ChromaDB
# (One-time per app run; no upload UI)
# =========================
def _pdf_to_text_from_path(pdf_path: Path) -> str:
    """Read a local PDF file into plain text."""
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
    """Simple char-based chunker with overlap to keep context."""
    text = text.replace("\r", "")
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_chars, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = max(end - overlap, 0)
    return chunks

def _find_local_pdfs(root_dir: str):
    """Find all PDFs under root_dir (recursively)."""
    pattern = str(Path(root_dir) / "**" / "*.pdf")
    return [Path(p) for p in glob.glob(pattern, recursive=True)]

def get_or_create_lab4_vdb_from_local(pdf_dir: str) -> chromadb.api.models.Collection.Collection:
    """
    Build (or return existing) ChromaDB collection 'Lab4Collection' with OpenAI embeddings.
    - Reads local PDFs from `pdf_dir`, converts to text, chunks, and inserts.
    - Adds 'doc_key' metadata for sorting/getting documents.
    - Stores the collection in st.session_state.Lab4_vectorDB
    - Only builds once per app run (checks session_state).
    """
    if "Lab4_vectorDB" in st.session_state and st.session_state.Lab4_vectorDB is not None:
        return st.session_state.Lab4_vectorDB

    # In-memory Chroma for this run only (cheap while the app stays live)
    client = chromadb.Client(Settings(anonymized_telemetry=False))
    embedder = OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-3-small"
    )

    collection = client.create_collection(
        name="Lab4Collection",
        embedding_function=embedder,
        metadata={"hnsw:space": "cosine"}
    )

    root = Path(pdf_dir)
    if not root.exists():
        st.warning(f"PDF directory '{pdf_dir}' not found. Skipping vector build.")
        st.session_state.Lab4_vectorDB = collection
        return collection

    pdf_paths = _find_local_pdfs(pdf_dir)
    if not pdf_paths:
        st.info(f"No PDFs found under '{pdf_dir}'.")
        st.session_state.Lab4_vectorDB = collection
        return collection

    for i, p in enumerate(pdf_paths):
        try:
            raw_text = _pdf_to_text_from_path(p)
        except Exception as e:
            st.warning(f"Could not read {p.name}: {e}")
            continue

        if not raw_text:
            st.warning(f"No text extracted from {p.name}. Skipping.")
            continue

        chunks = _chunk_text(raw_text, chunk_chars=2000, overlap=200)
        if not chunks:
            continue

        # Sortable key to group chunks from the same source
        # Include a zero-padded index so ordering is stable and sortable lexically
        doc_key = f"{i:04d}_{p.name}"

        ids = []
        docs = []
        metas = []
        for idx, chunk in enumerate(chunks):
            uid = f"{doc_key}_chunk_{idx}_{uuid.uuid4().hex[:8]}"
            ids.append(uid)
            docs.append(chunk)
            metas.append({
                "doc_key": doc_key,           # for grouping/sorting
                "source_path": str(p),        # full path
                "source_name": p.name,        # filename
                "chunk_index": idx,           # order within doc
                "total_chunks": len(chunks),
            })

        collection.add(ids=ids, documents=docs, metadatas=metas)

    st.session_state.Lab4_vectorDB = collection
    return collection

# Build the collection ONCE per run (no upload widget)
with st.sidebar:
    st.subheader("📚 Local Docs")
    st.caption(f"Loading PDFs from: `{PDF_DIR}`")
    if "Lab4_vectorDB" not in st.session_state:
        with st.spinner("Building Lab4Collection from local PDFs (one-time this run)…"):
            vdb = get_or_create_lab4_vdb_from_local(PDF_DIR)
            st.success(f"Vector DB ready. Chunks: ~{vdb.count()}")

    # (Optional) small test/query UI
    vdb = st.session_state.get("Lab4_vectorDB")
    if vdb:
        st.divider()
        q = st.text_input("Quick search your docs")
        if q:
            res = vdb.query(query_texts=[q], n_results=5)
            st.write(res)

# =========================
# Chat history display
# =========================
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

# =========================
# Chat input & streaming
# =========================
prompt = st.chat_input("Ask me anything…")
if prompt and prompt != st.session_state.last_handled_prompt:
    st.session_state.last_handled_prompt = prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    limited_history = trim_messages_to_tokens(
        st.session_state.messages, max_tokens=200, model=model_name
    )
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
