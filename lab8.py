import os
import re
import io
import sys
import json
import math
import time
import pdfplumber
import streamlit as st
from typing import List, Dict, Any, Tuple
from pathlib import Path

# --- Chroma on Streamlit Cloud needs sqlite shim ---
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI

# ---------------- UI ----------------
st.set_page_config(page_title="SEC 10-Q RAG Bot", page_icon="📊", layout="wide")
st.title("📊 IST 688 — SEC 10-Q RAG + Re-Ranking Chatbot")

# ---------------- Secrets / Clients ----------------
def _secret(name: str) -> str:
    try:
        return st.secrets[name].strip()
    except KeyError:
        return ""

OPENAI_KEY = _secret("OPENAI_API_KEY")
if not OPENAI_KEY:
    st.error("Missing `OPENAI_KEY` in Streamlit secrets.")
    st.stop()

oc = OpenAI(api_key=OPENAI_API_KEY, timeout=60, max_retries=2)

# ---------------- Paths / Constants ----------------
PERSIST_DIR = "sec10q_chroma"
COLLECTION  = "sec10q_collection"
MAX_CHUNK_CHARS = 1800   # for display + prompt safety
CHUNK_SIZE = 1400        # characters
CHUNK_OVERLAP = 200      # characters

Path(PERSIST_DIR).mkdir(parents=True, exist_ok=True)

# ---------------- Helpers ----------------
def _clean_text(t: str) -> str:
    t = re.sub(r"\s+", " ", t or " ")
    return t.strip()

def _chunk_text(text: str, company: str, source_name: str, page_no: int) -> List[Dict[str, Any]]:
    text = text.replace("\r", "")
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + CHUNK_SIZE, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({
                "content": chunk,
                "metadata": {
                    "company": company,
                    "source": source_name,
                    "page": page_no
                }
            })
        if end == n:
            break
        start = max(end - CHUNK_OVERLAP, 0)
    return chunks

def _extract_pdf_chunks(file: io.BytesIO, filename: str, company_hint: str = "") -> List[Dict[str, Any]]:
    """Extract per-page text via pdfplumber and chunk."""
    chunks = []
    company = company_hint or (
        "Amazon" if "amazon" in filename.lower() or "amzn" in filename.lower() else
        "Apple"  if "apple"  in filename.lower() or "aapl" in filename.lower() else
        "Unknown"
    )
    with pdfplumber.open(file) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            txt = _clean_text(txt)
            if not txt:
                continue
            page_chunks = _chunk_text(txt, company, filename, i)
            chunks.extend(page_chunks)
    return chunks

# ---------------- Vector DB (Chroma) ----------------
@st.cache_resource(show_spinner=True)
def _get_collection():
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    embedder = OpenAIEmbeddingFunction(api_key=OPENAI_KEY, model_name="text-embedding-3-small")
    col = client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=embedder,
        metadata={"hnsw:space": "cosine"}
    )
    return col

collection = _get_collection()

def _already_indexed_ids():
    try:
        got = collection.get()
        return set(got.get("ids", []))
    except Exception:
        return set()

def _add_chunks_to_chroma(doc_chunks: List[Dict[str, Any]], doc_key: str):
    existing = _already_indexed_ids()
    ids, docs, metas = [], [], []
    added = 0
    for idx, ch in enumerate(doc_chunks):
        cid = f"{doc_key}_{idx:04d}"
        if cid in existing:
            continue
        ids.append(cid)
        docs.append(ch["content"])
        metas.append(ch["metadata"])
        added += 1
    if ids:
        collection.add(ids=ids, documents=docs, metadatas=metas)
    return added

# ---------------- Sidebar: Upload & Controls ----------------
with st.sidebar:
    st.header("Step 1 — Upload 10-Q PDFs")
    uploads = st.file_uploader("Upload Amazon / Apple 10-Q PDFs", type=["pdf"], accept_multiple_files=True)
    company_override = st.text_input("Optional company override (applies to *all* new uploads)", value="")
    if st.button("Process PDFs"):
        if not uploads:
            st.warning("Please upload at least one PDF.")
        else:
            total_added = 0
            for up in uploads:
                chunks = _extract_pdf_chunks(up, up.name, company_override.strip())
                added = _add_chunks_to_chroma(chunks, doc_key=up.name)
                total_added += added
            st.success(f"Indexed {total_added} chunks. (New only; duplicates are skipped.)")

    st.header("Retrieval")
    k_retrieve = st.slider("Max chunks to retrieve", 3, 30, 8, step=1)
    use_llm_rerank = st.checkbox("Use LLM re-ranking (reorder retrieved chunks)", value=False)
    company_filter = st.multiselect("Filter by company (optional)", ["Amazon", "Apple", "Unknown"])

# ---------------- Retrieval ----------------
def retrieve_chunks(query: str, n_results: int, company_filter: List[str]) -> List[Dict[str, Any]]:
    res = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    results = []
    for text, meta, dist in zip(docs, metas, dists):
        if company_filter and meta.get("company") not in company_filter:
            continue
        results.append({
            "content": text,
            "metadata": meta,
            "distance": dist
        })
    return results

def llm_rerank(query: str, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ask the LLM to score each chunk for relevance (1–10), return reordered list.
    JSON-only response; robust parsing.
    """
    pack = []
    for i, it in enumerate(items):
        meta = it["metadata"]
        pack.append({
            "id": i,
            "company": meta.get("company",""),
            "source": meta.get("source",""),
            "page": meta.get("page", 0),
            "snippet": it["content"][:MAX_CHUNK_CHARS],
        })

    instr = (
        "You are ranking 10-Q text chunks for their relevance to the user's query.\n"
        "Score each item from 1 (not relevant) to 10 (highly relevant). "
        "Prefer chunks that directly discuss the user's query and contain concrete numbers or MD&A content. "
        "Return STRICT JSON: [{id, score}]. No extra text."
    )
    prompt = [
        {"role":"system","content": instr},
        {"role":"user","content": f"Query: {query}\n\nChunks:\n{json.dumps(pack)}"}
    ]
    try:
        resp = oc.chat.completions.create(
            model="gpt-4o-mini",
            messages=prompt,
            temperature=0.2
        )
        txt = resp.choices[0].message.content.strip()
        m = re.search(r"\[.*\]", txt, flags=re.DOTALL)
        arr = json.loads(m.group(0) if m else txt)
        by_id = {int(a["id"]): float(a.get("score", 5)) for a in arr if "id" in a}
    except Exception:
        by_id = {i: 5.0 for i in range(len(items))}

    # Attach score and sort desc
    ranked = []
    for i, it in enumerate(items):
        it2 = dict(it)
        it2["_llm_score"] = by_id.get(i, 5.0)
        ranked.append(it2)
    ranked.sort(key=lambda x: x.get("_llm_score", 0), reverse=True)
    return ranked

# ---------------- Answer Generation ----------------
SYSTEM_PROMPT = (
    "You are a helpful financial analysis assistant for SEC 10-Q filings. "
    "Use ONLY the provided CONTEXT chunks when possible. "
    "If answering, cite sources as [Company, page N]. "
    "Be concise and neutral; include key figures if present."
)

def answer_with_context(query: str, context_items: List[Dict[str, Any]]) -> str:
    # Build a compact context string
    ctx_blocks = []
    for it in context_items:
        meta = it["metadata"]
        tag = f"[{meta.get('company','?')}, page {meta.get('page','?')}]"
        ctx_blocks.append(f"{tag} {it['content'][:MAX_CHUNK_CHARS]}")
    context_text = "\n\n---\n\n".join(ctx_blocks)

    messages = [
        {"role":"system","content": SYSTEM_PROMPT},
        {"role":"system","content": f"CONTEXT:\n{context_text}"},
        {"role":"user","content": query}
    ]
    stream = oc.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        stream=True,
        temperature=0.2
    )
    return stream  # Streamlit can consume this

# ---------------- Chat UI ----------------
st.header("Step 2 — Ask Retrieval Questions")
default_q = "Summarize the financial performance this quarter."
user_q = st.text_input("Your question:", value=default_q, placeholder="e.g., What are the main risks?")
run = st.button("Run")

# A place to show retrieved sources
results_container = st.container()
answer_container = st.container()

if run:
    with st.spinner("Retrieving relevant chunks…"):
        raw_hits = retrieve_chunks(user_q, n_results=max(30, k_retrieve), company_filter=company_filter)
        # keep first k before optional re-rank (mimic “max chunks to retrieve”)
        hits = raw_hits[:k_retrieve]

        if use_llm_rerank and hits:
            hits = llm_rerank(user_q, hits)

    with results_container:
        st.subheader("Retrieved Chunks (top to bottom):")
        for i, it in enumerate(hits, start=1):
            meta = it["metadata"]
            tag = f"{meta.get('company','?')} — {meta.get('source','')}, p.{meta.get('page','?')}"
            cols = st.columns([0.1, 0.9])
            cols[0].markdown(f"**{i}.**")
            cols[1].markdown(f"**{tag}**")
            cols[1].write(it["content"][:MAX_CHUNK_CHARS] + ("…" if len(it["content"])>MAX_CHUNK_CHARS else ""))
            if "_llm_score" in it:
                st.caption(f"LLM relevance score: {it['_llm_score']:.1f}")

    with answer_container:
        st.subheader("Answer")
        with st.chat_message("assistant"):
            stream = answer_with_context(user_q, hits)
            final_text = st.write_stream(stream)

# ---------------- Step 3/4 Hints ----------------
with st.expander("💡 Tips for your Lab Write-Up"):
    st.markdown("""
- **Compare Across Companies**: Ask the *same* question after filtering “Company” in the sidebar or by uploading both filings. Note differences in MD&A tone, growth drivers, and risk language.
- **Explore Re-Ranking**: Toggle **LLM re-ranking** and vary **Max chunks to retrieve** (e.g., 5, 10, 20). Note how sources and answers change.
- **Screenshots**: Include your question, top chunks screenshot, and final answer with citations.
- **Limitations**: OCR/scanned PDFs may extract poorly; chunking can miss cross-page context; LLM re-ranking may occasionally over-prioritize generic risk language.
""")
