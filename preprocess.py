# preprocess.py

import PyPDF2
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

pdf_path = "/Users/adityameka/Downloads/MANUAL.pdf"


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"  # Add newline for separation between pages
    return text

def save_text_to_file(text, output_path):
    """Save extracted text to a plain text file."""
    with open(output_path, "w") as f:
        f.write(text)

def chunk_text(text, chunk_size=1000):
    """Chunk the text into smaller pieces."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def generate_embeddings(chunks):
    """Generate embeddings for each text chunk."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(chunks).astype("float32")  # Convert to float32 for FAISS

def create_faiss_index(embeddings):
    """Create a FAISS index from the embeddings."""
    embedding_dim = embeddings.shape[1]  # Number of dimensions in each embedding
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings)
    return index

def main():
    pdf_path = "uscis_manual.pdf"  # Update with the correct path to your PDF file
    output_text_path = "uscis_manual.txt"  # Path to save the extracted text
    faiss_index_path = "faiss_index.index"  # Path to save the FAISS index
    
    # Step 1: Extract text from PDF
    extracted_text = extract_text_from_pdf(pdf_path)
    
    # Step 2: Save the extracted text to a file
    save_text_to_file(extracted_text, output_text_path)
    print(f"Extracted text saved to {output_text_path}")

    # Step 3: Chunk the text
    chunks = chunk_text(extracted_text)
    print(f"Total chunks created: {len(chunks)}")

    # Step 4: Generate embeddings
    embeddings = generate_embeddings(chunks)
    print(f"Generated {len(embeddings)} embeddings.")

    # Step 5: Create and save FAISS index
    index = create_faiss_index(embeddings)
    faiss.write_index(index, faiss_index_path)
    print(f"FAISS index saved to {faiss_index_path}")

if __name__ == "__main__":
    main()
