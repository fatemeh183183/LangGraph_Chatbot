
import streamlit as st
from chatbot_core import graph
from dotenv import load_dotenv

load_dotenv() #Loads the GROQ_API_KEY from .env

st.set_page_config(page_title="LangGraph Chatbot", page_icon="🧠")
st.title("🧠 LangGraph Chatbot with Wikipedia & Arxiv")

user_input = st.text_input("Ask me anything:")

if user_input:
    with st.spinner("Thinking..."):
        events = graph.stream({"messages": [{"role": "user", "content": user_input}]}, stream_mode="values")
        for event in events:
            if "messages" in event:
                output = event["messages"][-1].content
                st.markdown(f"**🤖 Answer:** {output}")
