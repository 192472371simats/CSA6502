import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# -----------------------------------
# 1. Documents
# -----------------------------------

documents = [
    "Artificial intelligence is the simulation of human intelligence by machines.",
    "Machine learning is a branch of artificial intelligence that learns from data.",
    "Deep learning uses neural networks with multiple layers.",
    "Reinforcement learning learns through rewards and penalties."
]

# -----------------------------------
# 2. Load Embedding Model
# -----------------------------------

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# -----------------------------------
# 3. Generate Document Embeddings
# -----------------------------------

embeddings = embedding_model.encode(
    documents
)

embeddings = np.array(
    embeddings
).astype("float32")

print("Embeddings generated successfully.")

# -----------------------------------
# 4. Create FAISS Vector Database
# -----------------------------------

index = faiss.IndexFlatL2(
    embeddings.shape[1]
)

index.add(embeddings)

print("FAISS vector database created.")
print("Number of documents stored:", index.ntotal)

# -----------------------------------
# 5. Get User Question
# -----------------------------------

question = input("\nAsk a question: ")

# -----------------------------------
# 6. Generate Query Embedding
# -----------------------------------

query_embedding = embedding_model.encode(
    [question]
)

query_embedding = np.array(
    query_embedding
).astype("float32")

# -----------------------------------
# 7. Similarity Search
# -----------------------------------

k = 2

distances, indices = index.search(
    query_embedding,
    k
)

# -----------------------------------
# 8. Display Retrieved Documents
# -----------------------------------

print("\nRetrieved Context:")

for rank, idx in enumerate(
    indices[0],
    start=1
):
    print(f"{rank}. {documents[idx]}")

# -----------------------------------
# 9. Generate Answer
# -----------------------------------

best_index = indices[0][0]

answer = documents[best_index]

print("\nAnswer:")
print(answer)
