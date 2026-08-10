from google import genai

client = genai.Client(api_key="GEMINI_API_KEY")
# ----------------------------------
# Summarization Prompt
# ----------------------------------

summary_prompt = """
Summarize the following article in exactly 50 words.

Artificial Intelligence (AI) is transforming healthcare by improving disease diagnosis,
medical imaging, drug discovery, personalized treatment, virtual health assistants,
and patient monitoring. AI helps doctors make faster and more accurate decisions,
reduces medical errors, and improves healthcare accessibility.
"""

# ----------------------------------
# Email Generation Prompt
# ----------------------------------

email_prompt = """
Write a professional email requesting two days of leave due to illness.

Requirements:
- Subject
- Proper greeting
- Reason
- Leave dates
- Closing
"""

# ----------------------------------
# Content Generation Prompt
# ----------------------------------

content_prompt = """
Write a promotional social media post for an AI Workshop.

Requirements:
- Attractive title
- Maximum 80 words
- Include relevant hashtags
"""

# ----------------------------------
# Function
# ----------------------------------

def generate(prompt):

    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt
    )

    return response.text

# ----------------------------------
# Execute
# ----------------------------------

print("="*70)
print("SUMMARIZATION")
print("="*70)
print(generate(summary_prompt))

print("\n")

print("="*70)
print("EMAIL GENERATION")
print("="*70)
print(generate(email_prompt))

print("\n")

print("="*70)
print("CONTENT GENERATION")
print("="*70)
print(generate(content_prompt))
