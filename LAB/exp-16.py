from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Python is a popular programming language.",
    "Machine learning allows computers to learn from data.",
    "Football is a popular sport.",
    "Artificial intelligence is used in many applications."
]

query = "What is machine learning?"

# Generate embeddings
doc_embeddings = model.encode(documents)
query_embedding = model.encode([query])

# Calculate similarity
similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

# Display results
for i, score in enumerate(similarities):
    print(f"{documents[i]} -> Similarity: {score:.4f}")

best_index = similarities.argmax()

print("\nMost Similar Document:")
print(documents[best_index])
