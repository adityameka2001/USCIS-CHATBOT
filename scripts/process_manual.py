import langchain 
import torch
import faiss
import numpy as np

with open("Policy_Manual_USCIS.txt", "r") as f:
    manual_content = f.read()

from langchain.text_splitter import RecursiveCharacterTextSplitter

# Split the manual into chunks of size 500 characters with 50-character overlap
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
manual_chunks = splitter.split_text(manual_content)
print(f"Total chunks created: {len(manual_chunks)}")

from transformers import AutoModel, AutoTokenizer


# Load a sentence-transformer model from Hugging Face
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

# Generate embeddings for all chunks
embeddings = [embed_text(chunk) for chunk in manual_chunks]
print("Embeddings generated for all chunks.")


# Determine the dimension of the embeddings
embedding_dim = embeddings[0].shape[1]

# Create a FAISS index with L2 (Euclidean) distance metric
index = faiss.IndexFlatL2(embedding_dim)

# Add the embeddings to the index
index.add(np.array(embeddings))
print(f"Total documents indexed: {index.ntotal}")

faiss.write_index(index, "uscis_manual_index.faiss")
print("Index saved as 'uscis_manual_index.faiss'.")

index = faiss.read_index("uscis_manual_index.faiss")

query_embedding = embed_text("How to apply for citizenship?")
distances, indices = index.search(np.array([query_embedding]), k=3)

print("Top 3 relevant chunks:")
for idx in indices[0]:
    print(manual_chunks[idx])


