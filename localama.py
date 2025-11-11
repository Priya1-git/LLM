from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

import streamlit as st
import os
#from dotenv import load_dotenv
#load_dotenv()

os.environ["OPENAI_API_KEY"] = "sk-bXb1pHpW87M8rzbid9asdfasfG4T3BlbasdfOR4ocCCruyiOPYA"

os.environ["LANGCHAIN_TRACING_V2"]="true"

os.environ["LANGCHAIN_API_KEY"]= "lsv2_sk_fa97asdfc99d8d3564675sdgfdsdb7db0ba_d91f667dd0"


#os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

## Prompt Template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","I am chatbot. I am hear to assist you. Please type your queries"),
        ("user","Question:{question}")
    ]
)
## streamlit framework

st.title('LLM WITH OLLAMA API')
input_text=st.text_input("I am chatbot. How may I help you")

# ollama LLAma2 LLm 
llm=Ollama(model="llama2")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))