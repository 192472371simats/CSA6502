import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. Document Loading
# ==========================================

document = """
Artificial intelligence is a field of computer science.
Machine learning is a subset of artificial intelligence.
Machine learning allows computers to learn from data.
Deep learning uses multi-layer neural networks.
Reinforcement learning uses rewards and penalties.
Natural language processing enables computers to understand human language.
"""

print("Document loaded successfully.")

# ==========================================
# 2. Text Chunking
# ==========================================

chunk_size = 100

chunks = [
    document[i:i + chunk_size]
    for i in range(0, len(document), chunk_size)
]

print("Number of chunks:", len(chunks))

# ==========================================
# 3. Load Embedding Model
# ==========================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# ==========================================
# 4. Generate Embeddings
# ==========================================

embeddings = embedding_model.encode(chunks)

embeddings = np.array(
    embeddings
).astype("float32")

print("Embeddings generated.")

# ==========================================
# 5. Create FAISS Vector Database
# ==========================================

index = faiss.IndexFlatL2(
    embeddings.shape[1]
)

index.add(embeddings)

print("Vectors stored in FAISS:", index.ntotal)

# ==========================================
# 6. Get User Question
# ==========================================

question = input("\nEnter your question: ")

# ==========================================
# 7. Generate Query Embedding
# ==========================================

query_embedding = embedding_model.encode(
    [question]
)

query_embedding = np.array(
    query_embedding
).astype("float32")

# ==========================================
# 8. Retrieve Top-K Chunks
# ==========================================

k = min(3, len(chunks))

distances, indices = index.search(
    query_embedding,
    k
)

# ==========================================
# 9. Display Retrieved Context
# ==========================================

print("\nRetrieved Context:")

for rank, idx in enumerate(
    indices[0],
    start=1
):
    print(f"\nChunk {rank}:")
    print(chunks[idx].strip())

# ==========================================
# 10. Select Best Answer
# ==========================================

best_index = indices[0][0]

answer = chunks[best_index].strip()

# ==========================================
# 11. Display Final Answer
# ==========================================

print("\nFinal Answer:")
print(answer)
