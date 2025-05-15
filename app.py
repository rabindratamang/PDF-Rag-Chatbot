import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def split_text(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

def create_vectorstore(chunks):
    embeddings = OpenAIEmbeddings(openai_api_type=OPENAI_API_KEY)
    return FAISS.from_texts(chunks, embedding=embeddings)

def create_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0, model="gpt-4o-mini")
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)


st.set_page_config(page_title="PDF RAG Chatbot", layout="centered")
st.title("chat with your PDF (RAG)")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with st.spinner("Reading and indexing PDF..."):
        raw_text = extract_text_from_pdf(uploaded_file)
        chunks = split_text(raw_text)
        vectorstore = create_vectorstore(chunks)
        qa_chain = create_qa_chain(vectorstore)
    st.success("PDF loaded and ready!")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    query = st.text_input("Ask a question about the PDF")