import os
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer


# ==========================================
# 1. Document Folder
# ==========================================

folder = "documents"


if not os.path.exists(folder):

    print("Error: documents folder not found.")

    print(
        "Create a folder named 'documents' "
        "in the same folder as this program."
    )

    exit()


# ==========================================
# 2. Storage
# ==========================================

documents = []
sources = []


# ==========================================
# 3. Load Multiple Documents
# ==========================================

for filename in os.listdir(folder):

    if filename.endswith(".txt"):

        path = os.path.join(
            folder,
            filename
        )

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            text = file.read()

        # Split into sentences
        chunks = [
            sentence.strip()
            for sentence in text.split(".")
            if sentence.strip()
        ]

        for chunk in chunks:

            documents.append(chunk)

            sources.append(filename)


print("=================================")
print("   MULTI-DOCUMENT AI ASSISTANT")
print("=================================")

print(
    "Documents loaded:",
    len(set(sources))
)

print(
    "Total chunks:",
    len(documents)
)


# ==========================================
# 4. Embedding Model
# ==========================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ==========================================
# 5. Generate Embeddings
# ==========================================

embeddings = embedding_model.encode(
    documents
)

embeddings = np.array(
    embeddings
).astype("float32")

print("Embeddings generated.")


# ==========================================
# 6. FAISS Vector Database
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
# 7. AI Assistant
# ==========================================

print("\n=================================")
print("       AI ASSISTANT READY")
print("=================================")

print("Ask questions about your documents.")
print("Type 'exit' to stop.\n")


while True:

    question = input("Question: ").strip()

    if question.lower() == "exit":

        print("Assistant: Goodbye!")

        break


    # ======================================
    # Query Embedding
    # ======================================

    query_embedding = embedding_model.encode(
        [question]
    )

    query_embedding = np.array(
        query_embedding
    ).astype("float32")


    # ======================================
    # Top-3 Retrieval
    # ======================================

    k = min(
        3,
        len(documents)
    )

    distances, indices = index.search(
        query_embedding,
        k
    )


    # ======================================
    # Display Retrieved Documents
    # ======================================

    print("\nRetrieved Documents:")

    for rank, idx in enumerate(
        indices[0],
        start=1
    ):

        print(
            f"{rank}. [{sources[idx]}] "
            f"{documents[idx]}"
        )


    # ======================================
    # Best Answer
    # ======================================

    best_index = indices[0][0]

    answer = documents[best_index]

    print("\nAI Assistant:")
    print(answer)
