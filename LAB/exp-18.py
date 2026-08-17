import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Python is a programming language.",
    "Machine learning learns patterns from data.",
    "Artificial intelligence simulates human intelligence.",
    "Football is played between two teams.",
    "Deep learning uses artificial neural networks."
]

# Generate embeddings
embeddings = model.encode(documents)

# Convert to NumPy
embeddings = np.array(embeddings).astype("float32")

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# Add documents
index.add(embeddings)

# Query
query = input("Enter query: ")
query_embedding = model.encode([query])
query_embedding = np.array(query_embedding).astype("float32")

# Retrieve top 3
distances, indices = index.search(query_embedding, 3)

print("\nTop 3 Results:\n")

for i, idx in enumerate(indices[0]):
    print(f"Rank {i+1}")
    print("Document:", documents[idx])
    print("Distance:", distances[0][i])
    print()
