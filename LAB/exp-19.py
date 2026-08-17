import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Artificial intelligence is transforming healthcare.",
    "Machine learning is used for prediction and classification.",
    "Deep learning is based on neural networks.",
    "Reinforcement learning learns through rewards.",
    "Natural language processing handles human language.",
    "Computer vision processes images and videos."
]

# Create embeddings
embeddings = model.encode(documents)

# Convert to float32
embeddings = np.array(embeddings).astype("float32")

# Create vector database
index = faiss.IndexFlatL2(embeddings.shape[1])

# Store vectors
index.add(embeddings)

query = input("Enter your question: ")

query_embedding = model.encode([query])
query_embedding = np.array(query_embedding).astype("float32")

k = 3

distances, indices = index.search(query_embedding, k)

print("\nTop-K Retrieved Documents\n")

for rank, idx in enumerate(indices[0], start=1):
    print(f"{rank}. {documents[idx]}")
    print(f"Distance: {distances[0][rank-1]:.4f}\n")
