from google import genai

client = genai.Client(api_key="GEMINI_API_KEY")

# ----------------------------
# Zero-shot Prompt
# ----------------------------
zero_prompt = """
Generate a professional product description for a Smart Fitness Watch.
Keep the description within 80 words.
"""

# ----------------------------
# One-shot Prompt
# ----------------------------
one_prompt = """
Example:

Product: Wireless Bluetooth Earbuds

Description:
Compact wireless earbuds with Bluetooth 5.3, active noise cancellation,
crystal-clear sound quality, and up to 30 hours of battery life.

Now generate a similar product description for a Smart Fitness Watch.
Keep the response within 80 words.
"""

# ----------------------------
# Few-shot Prompt
# ----------------------------
few_prompt = """
Example 1

Product: Wireless Mouse

Description:
An ergonomic wireless mouse with adjustable DPI,
silent clicks, and long battery life.

Example 2

Product: Bluetooth Speaker

Description:
A portable Bluetooth speaker featuring deep bass,
waterproof design, and fast charging support.

Now generate a similar product description for a Smart Fitness Watch.
Keep the response within 80 words.
"""

# ----------------------------
# Function
# ----------------------------
def generate(prompt):
    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt
    )
    return response.text

# ----------------------------
# Execute All Prompts
# ----------------------------
print("=" * 70)
print("ZERO-SHOT PROMPT")
print("=" * 70)
print(generate(zero_prompt))

print("\n")

print("=" * 70)
print("ONE-SHOT PROMPT")
print("=" * 70)
print(generate(one_prompt))

print("\n")

print("=" * 70)
print("FEW-SHOT PROMPT")
print("=" * 70)
print(generate(few_prompt))
