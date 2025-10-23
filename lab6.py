# lab6.py — LangChain ReAct Research Assistant (papers via CSV + Chroma)

import os
import re
import sys
import pandas as pd
import streamlit as st
from typing import Tuple, List

# --- SQLite shim required by Chroma on Streamlit Cloud ---
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

# --------------------------- Page ---------------------------
st.set_page_config(page_title="Lab 6 — ReAct Research Assistant", page_icon="🔎")
st.title("🔎 Lab 6 — LangChain ReAct Research Assistant")

# --------------------------- Secrets ------------------------
def _secret(name: str, fallback: str = "") -> str:
    try:
        return st.secrets[name].strip().replace("\r", "").replace("\n", "")
    except KeyError:
        return fallback

# Support either "OPENAI_API_KEY" or "openai_api_key" (per assignment text)
OPENAI_KEY = _secret("OPENAI_API_KEY") or _secret("openai_api_key")
if not OPENAI_KEY:
    st.error("Missing OpenAI API key in secrets (OPENAI_API_KEY or openai_api_key).")
    st.stop()

# --------------------------- Sidebar ------------------------
st.sidebar.header("LLM")
model_name = st.sidebar.selectbox(
    "OpenAI model",
    ["gpt-4o-mini", "gpt-4o"],
    index=0,
)
st.sidebar.caption("Tip: 4o-mini is cheaper; 4o is stronger.")

# --------------------------- Session State -------------------
if "lab6_vectorstore" not in st.session_state:
    st.session_state.lab6_vectorstore = None
if "lab6_df" not in st.session_state:
    st.session_state.lab6_df = None
if "lab6_agent" not in st.session_state:
    st.session_state.lab6_agent = None
if "lab6_messages" not in st.session_state:
    st.session_state.lab6_messages = []  # [{"role": "user"/"assistant", "content": "..."}]

# --------------------------- Vectorstore ---------------------
@st.cache_resource(show_spinner=True)
def initialize_vectorstore() -> Tuple[Chroma, pd.DataFrame]:
    """
    Initialize persistent Chroma vector DB from local CSV of arXiv papers.

    Expects file in same directory:
      arxiv_papers_extended.csv
    with columns (flexible but recommended):
      title, authors, abstract, year, category, venue, link
    """
    CSV_PATH = "papers.csv"
    PERSIST_DIR = "LAB6_vector_db"

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"CSV '{CSV_PATH}' not found. Place your dataset CSV in the same folder as lab6.py."
        )

    os.makedirs(PERSIST_DIR, exist_ok=True)
    df = pd.read_csv(CSV_PATH)

    # Build LangChain Documents with useful metadata for the tools
    docs: List[Document] = []
    for _, row in df.iterrows():
        title = str(row.get("title", ""))
        authors = str(row.get("authors", ""))
        abstract = str(row.get("abstract", ""))
        year = str(row.get("year", ""))
        category = str(row.get("category", ""))
        venue = str(row.get("venue", ""))
        link = str(row.get("link", "")) or str(row.get("url", ""))

        text = (
            f"Title: {title}\n"
            f"Authors: {authors}\n"
            f"Abstract: {abstract}\n"
            f"Year: {year}\n"
            f"Category: {category}\n"
            f"Venue: {venue}\n"
            f"Link: {link}"
        )
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "title": title,
                    "authors": authors,
                    "year": year,
                    "category": category,
                    "venue": venue,
                    "link": link,
                },
            )
        )

    embeddings = OpenAIEmbeddings(api_key=OPENAI_KEY)
    # Create or load persistent Chroma store
    vectorstore = Chroma.from_documents(
        docs, embeddings, persist_directory=PERSIST_DIR
    )
    return vectorstore, df


# Initialize vectorstore and data (exactly once)
if st.session_state.lab6_vectorstore is None:
    try:
        vs, df = initialize_vectorstore()
        st.session_state.lab6_vectorstore = vs
        st.session_state.lab6_df = df
        st.success(f"Vector DB ready. Loaded {len(df)} papers.")
    except Exception as e:
        st.error(f"Vectorstore Error: {e}")
        st.stop()

# --------------------------- Custom Tools ---------------------
def search_papers(query: str) -> str:
    """Find research papers on a topic using semantic similarity search (top 5)."""
    vs: Chroma = st.session_state.lab6_vectorstore
    results = vs.similarity_search(query, k=5)
    if not results:
        return f"No papers found about '{query}'."
    out_lines = []
    for i, doc in enumerate(results):
        m = doc.metadata or {}
        out_lines.append(
            f"{i+1}. {m.get('title','(no title)')}\n"
            f"Authors: {m.get('authors','')}\n"
            f"Year: {m.get('year','')}\n"
            f"Venue: {m.get('venue','')}\n"
            f"Link: {m.get('link','')}\n"
        )
    return "\n".join(out_lines)

def compare_papers(query: str) -> str:
    """
    Compare two papers by title. Usage example:
      'Attention is All You Need and BERT: Pre-training of Deep Bidirectional...'
      or
      'Paper A vs. Paper B'
      'Paper A and Paper B'
    """
    parts = re.split(r"\s+and\s+|\s+vs\.?\s+", query, flags=re.IGNORECASE)
    if len(parts) < 2:
        return "Please specify two papers: 'paper1 and paper2' (or 'paper1 vs. paper2')."

    df = st.session_state.lab6_df

    def find_one(title_fragment: str):
        m = df[df["title"].astype(str).str.contains(title_fragment.strip(), case=False, na=False)]
        if m.empty:
            return None
        r = m.iloc[0]
        abstract = str(r.get("abstract", ""))[:600] + ("..." if len(str(r.get("abstract",""))) > 600 else "")
        return (
            f"Title: {r.get('title','')}\n"
            f"Authors: {r.get('authors','')}\n"
            f"Year: {r.get('year','')}\n"
            f"Venue: {r.get('venue','')}\n"
            f"Category: {r.get('category','')}\n"
            f"Abstract: {abstract}\n"
            f"Link: {r.get('link','') or r.get('url','')}\n"
        )

    p1, p2 = find_one(parts[0]), find_one(parts[1])
    if not p1 or not p2:
        return "Could not find one or both papers. Try using more of the titles."
    return f"### Paper 1\n{p1}\n### Paper 2\n{p2}"

tools = [
    Tool(
        name="SearchPapers",
        func=search_papers,
        description="Find research papers on a topic. Input: a topic or query string."
    ),
    Tool(
        name="ComparePapers",
        func=compare_papers,
        description="Compare two papers by (partial) titles. Input format: 'paper1 and paper2' or 'paper1 vs. paper2'."
    ),
]

# --------------------------- ReAct Agent -----------------------
# Short-term memory (conversation buffer)
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="output"
)

# LLM (OpenAI via LangChain)
llm = ChatOpenAI(model=model_name, api_key=OPENAI_KEY, temperature=0.2)

# Use canonical ReAct prompt from LangChain hub
prompt = hub.pull("hwchase17/react")

# Create agent + executor
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
st.session_state.lab6_agent = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True
)

# --------------------------- Chat UI ---------------------------
with st.sidebar:
    if st.button("Clear conversation"):
        st.session_state.lab6_messages = []
        st.rerun()

# Render prior turns
for m in st.session_state.lab6_messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

if user_input := st.chat_input("💬 Ask about research papers… (e.g., 'Transformers for vision', or 'Paper A vs. Paper B')"):
    # Show user turn
    st.session_state.lab6_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Invoke ReAct agent (tools available dynamically)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                response = st.session_state.lab6_agent.invoke({
                    "input": user_input,
                    "chat_history": st.session_state.lab6_messages
                })
                output = response.get("output", str(response))
            except Exception as e:
                output = f"(Agent error) {e}"
        st.write(output)

    # Save assistant turn
    st.session_state.lab6_messages.append({"role": "assistant", "content": output})

