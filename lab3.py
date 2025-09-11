import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Lab 3 — Streaming Chatbot", page_icon="💬")
st.title("💬 Lab 3 — Streaming Chatbot (with short-term memory)")

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

col1, col2 = st.columns(2)
with col1:
    model_name = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"], index=0)
with col2:
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

prompt = st.chat_input("Ask me anything…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=model_name,
            messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
            stream=True,
            timeout=60,
        )
        assistant_text = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": assistant_text})

# 4b) Call the LLM with the ENTIRE history (conversation buffer)
#     but truncate to only the last 2 messages for context
with st.chat_message("assistant"):
    # Get last two messages from memory
    last_two = st.session_state.messages[-2:] if len(st.session_state.messages) >= 2 else st.session_state.messages

    # Stream response
    stream = client.chat.completions.create(
        model=model_name,
        messages=[{"role": m["role"], "content": m["content"]} for m in last_two],
        stream=True,
        timeout=60,
    )
    assistant_text = st.write_stream(stream)

# 4c) Save assistant reply back into memory
st.session_state.messages.append({"role": "assistant", "content": assistant_text})

