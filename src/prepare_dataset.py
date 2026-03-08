import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Paths
DATA_PATH = "data/WELFake_Dataset.csv"
OUT_DIR = "data"

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

# Load dataset
print("📂 Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Make sure expected columns exist
print("✅ Columns in dataset:", df.columns.tolist())

# Combine title + text into a single column "content"
if "title" in df.columns and "text" in df.columns:
    df["content"] = (df["title"].fillna("") + " " + df["text"].fillna("")).str.strip()
elif "content" not in df.columns:
    raise ValueError("Dataset must have 'title'+'text' or 'content' column.")

# Train/test split
print("✂️ Splitting train/test...")
train, test = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label"]
)

# Save splits
train.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
test.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False)

print("✅ Train/Test split complete!")
print(f"Train size: {len(train)}, Test size: {len(test)}")
print("Files saved as data/train.csv and data/test.csv")
