from google import genai

client = genai.Client(api_key="GEMINI_API_KEY")

# ---------------------------------------
# Python Code Prompt
# ---------------------------------------

python_prompt = """
Generate only Python code.

Task:
Write a Python program to check whether a number is Prime.

Return only executable Python code.
Do not include explanations.
"""

# ---------------------------------------
# SQL Prompt
# ---------------------------------------

sql_prompt = """
Generate only SQL code.

Task:
Display all employees whose salary is greater than 50000.

Return only SQL query.
Do not include explanations.
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
# Execute
# ---------------------------------------

print("=" * 60)
print("GENERATED PYTHON CODE")
print("=" * 60)

print(generate(python_prompt))

print()

print("=" * 60)
print("GENERATED SQL QUERY")
print("=" * 60)

print(generate(sql_prompt))
