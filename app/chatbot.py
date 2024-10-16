import faiss
import pickle
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Load the FAISS index
def load_index(index_path):
    with open(index_path, "rb") as f:
        index = pickle.load(f)
    return index

index = load_index("../data/uscis_manual_index.faiss")

# Initialize retriever with FAISS
retriever = FAISS(index)

from langchain.llms import Ollama

# Load the LLaMA model (use Ollama or HuggingFace locally)
llm = Ollama(model="llama-3b", api_key="your_ollama_api_key")  # Example using Ollama

from langchain.chains import RetrievalQA

# Create the RAG-based chatbot chain
qa_chain = RetrievalQA(
    retriever=retriever,
    llm=llm,
    input_key="question"
)

def get_answer(question):
    try:
        response = qa_chain.run(question=question)
        return response
    except Exception as e:
        return f"Error: {str(e)}"

import streamlit as st

st.title("USCIS Manual Chatbot")
st.write("Ask any question related to the USCIS manual!")

user_input = st.text_input("Your Question:")

if user_input:
    answer = get_answer(user_input)
    st.write("Answer:", answer)
