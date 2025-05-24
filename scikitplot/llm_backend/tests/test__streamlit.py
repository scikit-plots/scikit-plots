"""Streamlit Test Example."""

import streamlit as st
from .. import get_backend

backend = get_backend("transformers", "mistralai/Mistral-7B-Instruct-v0.1")

st.title("Modular LLM Chat")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

user_input = st.text_input("Your message")

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    response = backend.chat(st.session_state["messages"])
    st.session_state["messages"].append({"role": "assistant", "content": response})

for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.markdown(f"**User:** {msg['content']}")
    else:
        st.markdown(f"**Assistant:** {msg['content']}")
