from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
import re

st.title("Priyanka Sulaganti bot using DeepSeek-R1")

# Prompt
prompt = ChatPromptTemplate.from_template(
    "Question: {question}\n\nAnswer: Let's think step by step."
)

# Model (ensure the model is pulled in Ollama: `ollama pull deepseek-r1`)
model = OllamaLLM(model="deepseek-r1")  # or "deepseek-r1:latest"

# Chain
chain = prompt | model

# UI
question = st.text_input("Enter your question here")

def strip_think(text: str) -> str:
    # DeepSeek-R1 sometimes returns <think>…</think> — hide it for users.
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)

if question:
    try:
        # Pass inputs as a dict when using LCEL pipelines
        response = chain.invoke({"question": question})
        st.write(strip_think(response))
    except Exception as e:
        st.error(f"Error: {e}")
