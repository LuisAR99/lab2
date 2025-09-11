# lab3.py  —  Streaming chatbot with short-term memory (LAB 3A)

import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Lab 3 — Streaming Chatbot", page_icon="💬")
st.title("💬 Lab 3 — Streaming Chatbot (with short-term memory)")

# 1) Get your key from Streamlit secrets (no text inputs in the UI)
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error(
        "Missing `OPENAI_API_KEY` in Streamlit secrets. "
        "Add it to .streamlit/secrets.toml or your cloud app’s Secrets."
    )
    st.stop()

client = OpenAI(api_key=api_key, timeout=60, max_retries=2)

# 2) Initialize the in-memory conversation buffer once per session
#    (LLMs are stateless; we keep state and pass it back each turn)
if "messages" not in st.session_state:
    st.session_state.messages = []  # each item: {"role": "user"|"assistant", "content": "..."}

# Optional: small toolbar
col1, col2 = st.columns(2)
with col1:
    model_name = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"], index=0)
with col2:
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# 3) Display the whole chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

# 4) Collect user input (chat style)
prompt = st.chat_input("Ask me anything…")
if prompt:
    # 4a) Show + store the user's message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # 4b) Call the LLM with the ENTIRE history (conversation buffer)
    #     and stream the response into the chat UI
    with st.chat_message("assistant"):
        # Stream response (write_stream returns the concatenated text)
        stream = client.chat.completions.create(
            model=model_name,
            messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
            stream=True,
            timeout=60,
        )
        assistant_text = st.write_stream(stream)

    # 4c) Save assistant reply back into memory so the next turn has context
    st.session_state.messages.append({"role": "assistant", "content": assistant_text})

