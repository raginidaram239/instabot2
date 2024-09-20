import pickle
import faiss
import numpy as np

# Load chunks and embeddings
with open('chunks.pkl', 'rb') as f:
    saved_chunks = pickle.load(f)
    
with open('embeddings.pkl', 'rb') as f:
    saved_embeddings = pickle.load(f)

# Check lengths
print(f"Number of chunks: {len(saved_chunks)}")
print(f"Number of embeddings: {len(saved_embeddings)}")

# Load FAISS index
index = faiss.read_index('faiss_index.bin')
print(f"FAISS index size: {index.ntotal}")

# Sample query to test FAISS index
query_vector = np.random.rand(1, saved_embeddings[0].shape[0]).astype('float32')
D, I = index.search(query_vector, k=5)
print(f"Sample query results: {I}, Distances: {D}")

# Optional: Verify saved embeddings against original
# Compare sample embeddings if needed
