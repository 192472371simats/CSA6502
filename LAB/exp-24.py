import os
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer


# ==========================================
# 1. Load External Document
# ==========================================

file_path = "knowledge.txt"

if not os.path.exists(file_path):

    print("Error: knowledge.txt not found.")
    print("Place knowledge.txt in the same folder as this program.")
    exit()

with open(
    file_path,
    "r",
    encoding="utf-8"
) as file:

    text = file.read()

print("External document loaded successfully.")


# ==========================================
# 2. Split Document into Chunks
# ==========================================

documents = [
    sentence.strip()
    for sentence in text.split(".")
    if sentence.strip()
]

print(
    "Number of chunks:",
    len(documents)
)


# ==========================================
# 3. Load Embedding Model
# ==========================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ==========================================
# 4. Generate Embeddings
# ==========================================

embeddings = embedding_model.encode(
    documents
)

embeddings = np.array(
    embeddings
).astype("float32")

print("Embeddings generated.")


# ==========================================
# 5. Create FAISS Database
# ==========================================

index = faiss.IndexFlatL2(
    embeddings.shape[1]
)

index.add(embeddings)

print(
    "Vectors stored in FAISS:",
    index.ntotal
)


# ==========================================
# 6. AI Assistant
# ==========================================

print("\n=================================")
print("      DOCUMENT AI ASSISTANT")
print("=================================")
print("Ask questions about the document.")
print("Type 'exit' to stop.\n")


while True:

    question = input("Question: ").strip()

    if question.lower() == "exit":

        print("Assistant: Goodbye!")

        break

    # --------------------------------------
    # Query Embedding
    # --------------------------------------

    query_embedding = embedding_model.encode(
        [question]
    )

    query_embedding = np.array(
        query_embedding
    ).astype("float32")

    # --------------------------------------
    # Similarity Search
    # --------------------------------------

    distances, indices = index.search(
        query_embedding,
        1
    )

    best_index = indices[0][0]

    answer = documents[best_index]

    # --------------------------------------
    # Display Result
    # --------------------------------------

    print("\nRetrieved Document:")
    print(answer)

    print("\nAssistant:")
    print(answer)
