# USCIS Manual Chatbot

## Overview
This chatbot system helps users get answers based on the USCIS manual using Python, FAISS, and Llama models.

## How to Run
1. Clone this repository

2. Install dependencies:
pip install -r requirements.txt

3. Process the USCIS manual:
process_manual.py

4. Run the application:
streamlit run app.py

* Use a virtual machine for increased efficiency as a good practice 

Technologies Used

Python 3.8+
FAISS: For vector search
LangChain: For language model integration
Llama 3.1: 8B parameter model
Streamlit: For UI

This chatbot leverages RAG techniques to provide relevant answers from the USCIS manual. By using **FAISS for efficient retrieval** and **Llama for generation**, the application ensures that users receive fast, context-aware responses through a user-friendly **Streamlit interface**.