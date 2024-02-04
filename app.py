import streamlit as st
from streamlit_chat import message
import tempfile
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import faiss
from langchain_community.llms import ctransformers
from langchain.chains import ConversationalRetrievalChain

DB_FAISS_PATH = "vectorstore/db_faiss"

def load_llm():
    llm = ctransformers(
        model = "llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type = "llama",
        max_new_tokens = 512,
        temperature = 0.5
    )
    return llm

st.title("Chat with CSV")
st.markdown("<h3 style='text-align: center; color: white;'> TEST </h3>", unsafe_allow_html=True)