# test_setup.py

try:
    import streamlit as st
    print("Streamlit imported successfully!")
except ImportError as e:
    print(f"Error importing Streamlit: {e}")

try:
    import faiss
    print("FAISS imported successfully!")
except ImportError as e:
    print(f"Error importing FAISS: {e}")

try:
    from sentence_transformers import SentenceTransformer
    print("Sentence Transformers imported successfully!")
except ImportError as e:
    print(f"Error importing Sentence Transformers: {e}")
