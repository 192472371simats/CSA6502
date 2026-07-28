
from huggingface_hub import InferenceClient

client = InferenceClient(
    api_key="YOUR_HUGGINGFACE_API_KEY"
)
# ---------------- ZERO-SHOT PROMPT ----------------
zero_prompt = """
Write a professional email requesting one day's leave due to illness in about 100 words.
"""

# ---------------- ONE-SHOT PROMPT ----------------
one_prompt = """
Example:

Subject: Leave Request

Dear Sir,

I request one day's leave to attend a family function. I will complete my pending work upon my return.

Thank you.

Now write a professional email requesting one day's leave due to illness.
Use a similar format and length.
"""

# ---------------- FEW-SHOT PROMPT ----------------
few_prompt = """
Example 1:

Subject: Leave Request

Dear Sir,

I request leave to attend a seminar. I will complete all pending work after returning.

Thank you.

Example 2:

Subject: Leave Request

Dear Sir,

I request leave due to a personal emergency. I will resume work as soon as possible.

Thank you.

Now write a professional email requesting one day's leave due to illness.
Use a similar format and length.
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

