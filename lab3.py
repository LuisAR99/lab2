import streamlit as st
from openai import OpenAI
import tiktoken  

st.set_page_config(page_title="Lab 3 — Streaming Chatbot", page_icon="💬")
st.title("💬 Lab 3 — Streaming Chatbot (token-limited context)")

try:
    api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error(
        "Missing `OPENAI_API_KEY` in Streamlit secrets. "
        "Add it to .streamlit/secrets.toml or your cloud app’s Secrets."
    )
    st.stop()

client = OpenAI(api_key=api_key, timeout=60, max_retries=2)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_handled_prompt" not in st.session_state:
    st.session_state.last_handled_prompt = None

# --- Helper to trim context to N tokens ---
def trim_messages_to_tokens(messages, max_tokens=200, model="gpt-4o-mini"):
    enc = tiktoken.encoding_for_model(model)
    result = []
    total = 0
    # Walk backwards (most recent first), collect until max_tokens
    for m in reversed(messages):
        tokens = len(enc.encode(m["content"]))
        if total + tokens > max_tokens:
            break
        result.append(m)
        total += tokens
    return list(reversed(result))  # reverse back to chronological order

col1, col2 = st.columns(2)
with col1:
    model_name = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"], index=0)
with col2:
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.last_handled_prompt = None
        st.rerun()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

prompt = st.chat_input("Ask me anything…")
if prompt and prompt != st.session_state.last_handled_prompt:
    st.session_state.last_handled_prompt = prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # 🔑 Trim context by token count (including this new user message)
    limited_context = trim_messages_to_tokens(st.session_state.messages, max_tokens=200, model=model_name)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=model_name,
            messages=limited_context,
            stream=True,
            timeout=60,
        )
        assistant_text = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": assistant_text})
    st.rerun()
