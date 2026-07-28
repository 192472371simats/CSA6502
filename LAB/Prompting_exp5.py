
from huggingface_hub import InferenceClient

client = InferenceClient(
    api_key="YOUR_HUGGINGFACE_API_KEY"
)

# ---------------- ZERO-SHOT PROMPT ----------------
zero_prompt = """
Generate a promotional social media post of about 50 words for an AI Workshop.
"""

# ---------------- ONE-SHOT PROMPT ----------------
one_prompt = """
Example:

Workshop: Python Programming

Post:
🚀 Join our Python Programming Workshop! Learn Python basics, coding techniques, and hands-on programming from industry experts. Register today and boost your programming skills!

Now generate a promotional social media post of about 50 words for an AI Workshop.
Keep the same style and length.
"""

# ---------------- FEW-SHOT PROMPT ----------------
few_prompt = """
Example 1:

Workshop: Data Science

Post:
📊 Join our Data Science Workshop! Learn data analysis, visualization, and machine learning through practical sessions. Register now to enhance your skills.

Example 2:

Workshop: Machine Learning

Post:
🤖 Explore Machine Learning through hands-on projects and expert guidance. Learn real-world applications and build intelligent solutions. Enroll today!

Now generate a promotional social media post of about 50 words for an AI Workshop.
Keep the same style and length as the examples.
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
