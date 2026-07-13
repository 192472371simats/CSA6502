import pandas as pd

# Load dataset
df = pd.read_csv(r"D:\CSA6502\students.csv")

# Display original dataset
print("Original Dataset:")
print(df)

# Fill missing values with the column average
df["Marks"] = df["Marks"].fillna(df["Marks"].mean())

# Display cleaned dataset
print("\nDataset after cleaning missing values:")
print(df)

# Statistical Information
print("\nAverage Marks:", df["Marks"].mean())
print("Highest Score:", df["Marks"].max())
