from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ==========================================
# 1. Python Knowledge Base
# ==========================================

documents = [
    "Python is a high-level programming language.",
    "Lists are ordered and mutable collections in Python.",
    "Tuples are ordered and immutable collections in Python.",
    "Dictionaries store data using key-value pairs in Python.",
    "Functions are reusable blocks of code in Python.",
    "A for loop is used to iterate over a sequence in Python.",
    "An if statement is used for conditional execution in Python.",
    "A class is a blueprint for creating objects in Python.",
    "Python supports object-oriented programming.",
    "Python uses indentation to define code blocks."
]


# ==========================================
# 2. Load Embedding Model
# ==========================================

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# ==========================================
# 3. Create FAISS Vector Database
# ==========================================

vector_db = FAISS.from_texts(
    documents,
    embeddings
)

print("Vector database created successfully.")


# ==========================================
# 4. Load FLAN-T5
# ==========================================

print("Loading language model...")

model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name
)

print("Language model loaded successfully.")


# ==========================================
# 5. Generate Answer
# ==========================================

def generate_answer(context, question):

    prompt = f"""
You are a Python programming tutor.

Use ONLY the information in the context to answer the question.

Context:
{context}

Question:
{question}

Give one short, direct answer.
Do not mention information that is not in the context.
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=False,
        num_beams=4
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return answer.strip()


# ==========================================
# 6. Greeting Detection
# ==========================================

greetings = [
    "hello",
    "hi",
    "hey",
    "good morning",
    "good afternoon",
    "good evening"
]


# ==========================================
# 7. Chatbot
# ==========================================

print("\n=================================")
print("       PYTHON AI CHATBOT")
print("=================================")
print("Ask questions about Python.")
print("Type 'exit' to stop.\n")


while True:

    question = input("You: ").strip()

    # --------------------------------------
    # Exit
    # --------------------------------------

    if question.lower() == "exit":
        print("Bot: Goodbye!")
        break


    # --------------------------------------
    # Greeting
    # --------------------------------------

    if question.lower() in greetings:

        print("Bot: Hello! I can answer questions about Python.")
        continue


    # --------------------------------------
    # Retrieve documents
    # --------------------------------------

    results = vector_db.similarity_search_with_score(
        question,
        k=1
    )


    best_document = results[0][0]
    distance = results[0][1]

    context = best_document.page_content


    # --------------------------------------
    # Display retrieved context
    # --------------------------------------

    print("\nRetrieved Context:")
    print(context)

    print("Similarity distance:", round(distance, 4))


    # --------------------------------------
    # Simple factual answers
    # --------------------------------------

    question_lower = question.lower()


    if "list" in question_lower and "python" in question_lower:

        answer = "Lists are ordered and mutable collections in Python."


    elif (
        "tuple" in question_lower
        or "tuples" in question_lower
    ):

        answer = "Tuples are ordered and immutable collections in Python."


    elif (
        "key-value" in question_lower
        or "key value" in question_lower
        or "dictionary" in question_lower
        or "dictionaries" in question_lower
    ):

        answer = "Dictionaries store data using key-value pairs in Python."


    elif "for loop" in question_lower:

        answer = "A for loop is used to iterate over a sequence in Python."


    elif "function" in question_lower:

        answer = "Functions are reusable blocks of code in Python."


    elif "class" in question_lower:

        answer = "A class is a blueprint for creating objects in Python."


    elif "indentation" in question_lower:

        answer = "Python uses indentation to define code blocks."


    else:

        answer = generate_answer(
            context,
            question
        )


    # --------------------------------------
    # Display Answer
    # --------------------------------------

    print("\nBot:", answer)
    print()
