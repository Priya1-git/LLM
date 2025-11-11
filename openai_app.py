from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
#from dotenv import load_dotenv

os.environ["OPENAI_API_KEY"] = "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"
#os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

## Langmith tracking
os.environ["LANGCHAIN_TRACING_V2"]="true"

os.environ["LANGCHAIN_API_KEY"]= "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"
#os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")


## Prompt Template

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","I am chatbot. I am hear to assist you. Please type your queries"),
        ("user","Question:{question}")
    ]
)

## streamlit framework

st.title('LLM-OPENAI PROJECT')
input_text=st.text_input("How may I help you")

# openAI LLm 
llm=ChatOpenAI(model="gpt-3.5-turbo")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({'question':input_text}))