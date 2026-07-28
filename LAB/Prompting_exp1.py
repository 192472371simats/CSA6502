from huggingface_hub import InferenceClient

client = InferenceClient(
    api_key="YOUR_HUGGINGFACE_API_KEY"
)

# ---------------- ZERO-SHOT PROMPT ----------------
zero_prompt = """
Generate a product description for a Smart Fitness Watch.
"""

# ---------------- ONE-SHOT PROMPT ----------------
one_prompt = """
Example:

Product: Wireless Bluetooth Earbuds

Description:
Compact wireless earbuds with noise cancellation, long battery life, and crystal-clear sound.

Now generate a product description for a Smart Fitness Watch.
"""

# ---------------- FEW-SHOT PROMPT ----------------
few_prompt = """
Example 1:

Product: Wireless Mouse

Description:
An ergonomic wireless mouse with adjustable DPI and long battery life.

Example 2:

Product: Bluetooth Speaker

Description:
A portable Bluetooth speaker with deep bass and waterproof design.

Now generate a product description for a Smart Fitness Watch.
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
