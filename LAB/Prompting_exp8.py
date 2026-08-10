from google import genai

client = genai.Client(api_key="GEMINI_API_KEY")
print("=" * 60)
print("GOOGLE GEMINI API INTEGRATION")
print("=" * 60)

# ---------------------------------------
# Get User Prompt
# ---------------------------------------

user_prompt = input("\nEnter your prompt: ")

# ---------------------------------------
# Generate Response
# ---------------------------------------

try:

    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=user_prompt
    )

    print("\n" + "=" * 60)
    print("AI RESPONSE")
    print("=" * 60)

    print(response.text)

except Exception as e:

    print("\nError occurred:")
    print(e)
