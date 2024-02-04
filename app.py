import streamlit as st
from streamlit_chat import message
import tempfile
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import faiss
from langchain_community.llms import ctransformers
from langchain.chains import ConversationalRetrievalChain

DB_FAISS_PATH = "sample/vectorstore/db_faiss"

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

uploaded_file = st.sidebar._file_uploader("Upload you data here", type="csv")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={
        'delimiter': ','
    })
    data = loader.load()
    st.json(data)
    embeddings = HuggingFaceEmbeddings(
        model_name = 'sentence-transformers/all-MiniLM-L6-v2'

    )

    db = faiss.from_documents(data, embeddings)
    db.save_local(DB_FAISS_PATH)