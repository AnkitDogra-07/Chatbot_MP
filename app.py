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
    