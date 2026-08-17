from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ==========================================
# 1. Knowledge Base
# ==========================================

documents = [
    "Lists are ordered and mutable collections in Python.",
    "Tuples are ordered and immutable collections in Python.",
    "Dictionaries store data using key-value pairs in Python.",
    "Sets are unordered collections of unique elements in Python.",
    "Functions are reusable blocks of code in Python.",
    "Classes are blueprints for creating objects in Python."
]

# ==========================================
# 2. Load Embedding Model
# ==========================================

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ==========================================
# 3. Create Vector Database
# ==========================================

vector_db = FAISS.from_texts(
    documents,
    embeddings
)

print("Vector database created.")

# ==========================================
# 4. Conversation Memory
# ==========================================

conversation_history = []

last_topic = None

# ==========================================
# 5. Chatbot
# ==========================================

print("\n=================================")
print("     CONTEXT-AWARE CHATBOT")
print("=================================")
print("Type 'exit' to stop.\n")

while True:

    question = input("You: ").strip()

    if question.lower() == "exit":
        print("Bot: Goodbye!")
        break

    # --------------------------------------
    # Detect topic from current question
    # --------------------------------------

    question_lower = question.lower()

    topics = [
        "list",
        "tuple",
        "dictionary",
        "set",
        "function",
        "class"
    ]

    detected_topic = None

    for topic in topics:
        if topic in question_lower:
            detected_topic = topic
            break

    # --------------------------------------
    # Use previous topic for context
    # --------------------------------------

    if detected_topic is not None:

        last_topic = detected_topic

    elif last_topic is not None:

        question = last_topic + " " + question

    # --------------------------------------
    # Retrieve relevant document
    # --------------------------------------

    result = vector_db.similarity_search(
        question,
        k=1
    )

    answer = result[0].page_content

    # --------------------------------------
    # Handle context questions
    # --------------------------------------

    if (
        question_lower in ["is it mutable?", "is it mutable", "is it immutable?",
                           "is it immutable"]
        and last_topic == "list"
    ):
        answer = "Yes, lists are mutable."

    elif (
        question_lower in ["is it mutable?", "is it mutable"]
        and last_topic == "tuple"
    ):
        answer = "No, tuples are immutable."

    elif (
        question_lower in ["what about tuples?", "what about tuple?"]
    ):
        last_topic = "tuple"
        answer = "Tuples are ordered and immutable collections in Python."

    print("Bot:", answer)

    # --------------------------------------
    # Save conversation
    # --------------------------------------

    conversation_history.append(
        "User: " + question
    )

    conversation_history.append(
        "Bot: " + answer
    )
