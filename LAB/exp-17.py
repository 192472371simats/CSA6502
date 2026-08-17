from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Deep learning uses neural networks.",
    "Python is used for software development.",
    "Natural language processing deals with human language.",
    "Reinforcement learning uses rewards and penalties.",
    "Computer vision allows machines to understand images."
]

query = input("Enter your search query: ")

doc_embeddings = model.encode(documents)
query_embedding = model.encode([query])

similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

ranked_results = sorted(
    zip(documents, similarities),
    key=lambda x: x[1],
    reverse=True
)

print("\nSemantic Search Results:\n")

for document, score in ranked_results:
    print(f"Score: {score:.4f} | {document}")
