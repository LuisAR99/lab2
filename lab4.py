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
st.set_page_config(page_title="Lab 3 — Streaming Chatbot", page_icon="💬")
st.title("💬 Lab 3 — Streaming Chatbot (token-limited + kid-friendly)")

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
        for p
