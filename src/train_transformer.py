import pandas as pd
from datasets import Dataset
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizerFast, Trainer, TrainingArguments

# Load train/test data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# Use a smaller sample for faster demo training
train = train.sample(5000, random_state=42)   # pick 5k training samples
test = test.sample(1000, random_state=42)     # pick 1k test samples

print(f"✅ Using {len(train)} training samples and {len(test)} test samples for demo training.")


# Tokenizer
tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")

def encode(batch):
    return tokenizer(batch["content"], truncation=True, padding="max_length", max_length=256)

train_ds = Dataset.from_pandas(train)
test_ds = Dataset.from_pandas(test)

train_ds = train_ds.map(encode, batched=True)
test_ds = test_ds.map(encode, batched=True)

# Model
model = XLMRobertaForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=2)

# Training Args
training_args = TrainingArguments(
    output_dir="models/xlm_model",
  #  evaluation_strategy="epoch",
  #  save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    save_total_limit=1,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer
)

print("🚀 Training Transformer...")
trainer.train()

# Save model
model.save_pretrained("models/xlm_model")
tokenizer.save_pretrained("models/xlm_model")

print("✅ Transformer model fine-tuned and saved in models/xlm_model/")
