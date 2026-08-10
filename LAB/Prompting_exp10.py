from google import genai

client = genai.Client(api_key="GEMINI_API_KEY")
# ---------------------------------------
# Zero-shot Prompt
# ---------------------------------------

zero_prompt = """
Explain Machine Learning in about 60 words.
"""

# ---------------------------------------
# One-shot Prompt
# ---------------------------------------

one_prompt = """
Example

Artificial Intelligence:
Artificial Intelligence enables computers to perform tasks that usually require human intelligence.

Now explain Machine Learning in about 60 words.
"""

# ---------------------------------------
# Few-shot Prompt
# ---------------------------------------

few_prompt = """
Example 1

Artificial Intelligence:
Artificial Intelligence enables machines to imitate human intelligence.

Example 2

Deep Learning:
Deep Learning is a subset of Machine Learning that uses neural networks to solve complex problems.

Now explain Machine Learning in about 60 words.
"""

# ---------------------------------------
# Function
# ---------------------------------------

def generate(prompt):

    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt
    )

    return response.text

# ---------------------------------------
# Generate Responses
# ---------------------------------------

zero_output = generate(zero_prompt)
one_output = generate(one_prompt)
few_output = generate(few_prompt)

# ---------------------------------------
# Display Responses
# ---------------------------------------

print("="*70)
print("ZERO-SHOT OUTPUT")
print("="*70)
print(zero_output)

print("\n")

print("="*70)
print("ONE-SHOT OUTPUT")
print("="*70)
print(one_output)

print("\n")

print("="*70)
print("FEW-SHOT OUTPUT")
print("="*70)
print(few_output)

# ---------------------------------------
# Evaluation Prompt
# ---------------------------------------

evaluation_prompt = f"""
Compare the following three responses.

Response 1 (Zero-shot):
{zero_output}

Response 2 (One-shot):
{one_output}

Response 3 (Few-shot):
{few_output}

Evaluate them based on:

1. Accuracy
2. Completeness
3. Readability
4. Relevance

Finally,
- Identify the best prompting technique.
- Give reasons.
- Present the comparison in a table.
"""

evaluation = generate(evaluation_prompt)

print("\n")
print("="*70)
print("COMPARATIVE ANALYSIS")
print("="*70)
print(evaluation)
