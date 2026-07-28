from huggingface_hub import InferenceClient

client = InferenceClient(
    api_key="YOUR_HUGGINGFACE_API_KEY"
)

# ---------------- ZERO-SHOT PROMPT ----------------
zero_prompt = """
Write a blog of exactly 50 words on "Applications of Artificial Intelligence in Healthcare."
"""

# ---------------- ONE-SHOT PROMPT ----------------
one_prompt = """
Example:

Topic: Artificial Intelligence in Education

Blog:
Artificial Intelligence improves education through personalized learning, virtual tutors, and automated grading. It enhances learning experiences, supports teachers, and helps students achieve better academic performance in a smarter and more efficient way.

Now write a blog of exactly 50 words on "Applications of Artificial Intelligence in Healthcare."
Keep the same length and writing style as the example.
"""

# ---------------- FEW-SHOT PROMPT ----------------
few_prompt = """
Example 1:

Topic: Artificial Intelligence in Agriculture

Blog:
Artificial Intelligence helps farmers monitor crops, predict weather, detect diseases, and improve productivity while reducing costs and increasing agricultural efficiency.

Example 2:

Topic: Artificial Intelligence in Banking

Blog:
Artificial Intelligence enhances banking through fraud detection, customer support, risk analysis, and secure digital transactions, improving efficiency and customer satisfaction.

Now write a blog of exactly 50 words on "Applications of Artificial Intelligence in Healthcare."
Keep the same length and writing style as the examples.
"""

# ---------------- ZERO-SHOT OUTPUT ----------------
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": zero_prompt}],
    max_tokens=200
)

print("=" * 60)
print("ZERO-SHOT OUTPUT")
print("=" * 60)
print(response.choices[0].message.content)

# ---------------- ONE-SHOT OUTPUT ----------------
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": one_prompt}],
    max_tokens=200
)

print("\n" + "=" * 60)
print("ONE-SHOT OUTPUT")
print("=" * 60)
print(response.choices[0].message.content)

# ---------------- FEW-SHOT OUTPUT ----------------
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": few_prompt}],
    max_tokens=200
)

print("\n" + "=" * 60)
print("FEW-SHOT OUTPUT")
print("=" * 60)
print(response.choices[0].message.content)
