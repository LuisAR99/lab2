import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Lab 3 — Streaming Chatbot", page_icon="💬")
st.title("💬 Lab 3 — Streaming Chatbot (with short-term memory)")

# Secrets
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error(
        "Missing `OPENAI_API_KEY` in Streamlit secrets. "
        "Add it to .streamlit/secrets.toml or your cloud app’s Secrets."
    )
    st.stop()

client = OpenAI(api_key=api_key, timeout=60, max_retries=2)

# Short-term memory buffer
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": "user"|"assistant", "content": "..."}]

# Guard to prevent double-generating on reruns
if "last_handled_prompt" not in st.session_state:
    st.session_state.last_handled_prompt = None

# Controls
col1, col2 = st.columns(2)
with col1:
    model_name = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"], index=0)
with col2:
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.last_handled_prompt = None
        st.rerun()

# Render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

# New user input
prompt = st.chat_input("Ask me anything…")

# Only handle a prompt ONCE (prevents duplicate replies on rerun)
if prompt and prompt != st.session_state.last_handled_prompt:
    st.session_state.last_handled_prompt = prompt  # mark as handled

    # Append + display user msg
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Limit context to last two messages (sliding window)
    last_two = st.session_state.messages[-2:]  # includes just-added user turn

    # Stream assistant reply (display once, then persist)
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=model_name,
            messages=[{"role": m["role"], "content": m["content"]} for m in last_two],
            stream=True,
            timeout=60,
        )
        assistant_text = st.write_stream(stream)

    # Save reply to history and re-run so it renders once from history next time
    st.session_state.messages.append({"role": "assistant", "content": assistant_text})
    st.rerun()

