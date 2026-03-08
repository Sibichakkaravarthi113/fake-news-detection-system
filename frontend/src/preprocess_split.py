import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Ensure processed folder exists
os.makedirs("data/processed", exist_ok=True)

# Load raw dataset
df = pd.read_csv("data/raw/fake_news_dataset.csv")

# Split into train/test (80/20)
train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# Save
train.to_csv("data/processed/train.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)

print("✅ Train/test split created:")
print(f"Train size = {len(train)}, Test size = {len(test)}")
