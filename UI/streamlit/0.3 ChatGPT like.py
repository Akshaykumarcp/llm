import streamlit as st
import ollama

# ollama.list()

st.title("ChatGPT-like clone")
st.caption("🚀 A streamlit chatbot powered by Mistral LLM")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input():
    print(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = ollama.chat(
                    model='mistral',
                    messages=st.session_state.messages,
                    # stream=True,
                    )
    msg = response['message']['content']

    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)