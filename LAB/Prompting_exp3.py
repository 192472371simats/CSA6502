from huggingface_hub import InferenceClient

client = InferenceClient(
    api_key="YOUR_HUGGINGFACE_API_KEY"
)
article = """
Artificial Intelligence is transforming healthcare by improving disease diagnosis, drug discovery, robotic surgery, patient monitoring, and personalized treatment. It reduces medical errors, improves hospital management, enables faster decision-making, and enhances healthcare accessibility through telemedicine and virtual health assistants.
"""

# ---------------- ZERO-SHOT PROMPT ----------------
zero_prompt = f"""
Summarize the following article in exactly 50 words.

Article:
{article}
"""

# ---------------- ONE-SHOT PROMPT ----------------
one_prompt = f"""
Example:

Article:
Artificial Intelligence improves agriculture by monitoring crops, predicting weather, and detecting plant diseases.

Summary:
Artificial Intelligence helps farmers improve crop productivity through weather prediction, disease detection, and crop monitoring, enabling better farming decisions and increased agricultural efficiency.

Now summarize the following article in exactly 50 words.

Article:
{article}
"""

# ---------------- FEW-SHOT PROMPT ----------------
few_prompt = f"""
Example 1:

Article:
Artificial Intelligence improves education through personalized learning and automated grading.

Summary:
Artificial Intelligence enhances education by personalizing learning, automating grading, and improving student performance and teaching effectiveness.

Example 2:

Article:
Artificial Intelligence improves banking by detecting fraud and automating customer support.

Summary:
Artificial Intelligence strengthens banking by detecting fraud, automating customer support, improving financial security, and enhancing customer services.

Now summarize the following article in exactly 50 words.

Article:
{article}
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
