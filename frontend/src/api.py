# ============================================================
#  Fake News Detection API - Baselines + Transformer (XLM-RoBERTa)
# ============================================================

from fastapi import FastAPI
from pydantic import BaseModel
import joblib, torch
from transformers import XLMRobertaTokenizerFast, XLMRobertaForSequenceClassification
import numpy as np
import uvicorn
import os

# ------------------------------------------------------------
# Initialize FastAPI App
# ------------------------------------------------------------
app = FastAPI(
    title="Fake News Detection API",
    description="API combining baseline ML models and Transformer (XLM-RoBERTa) for fake news detection.",
    version="2.0"
)

# ------------------------------------------------------------
# Define Input Schema
# ------------------------------------------------------------
class NewsInput(BaseModel):
    text: str

# ------------------------------------------------------------
# Load Models
# ------------------------------------------------------------
print("📦 Loading baseline models...")
tfidf = joblib.load("models/tfidf.joblib")
lr = joblib.load("models/LogisticRegression.joblib")
nb = joblib.load("models/NaiveBayes.joblib")
svm = joblib.load("models/SVM.joblib")

print("🤖 Loading Transformer model...")
tokenizer = XLMRobertaTokenizerFast.from_pretrained("models/xlm_model")
transformer = XLMRobertaForSequenceClassification.from_pretrained("models/xlm_model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transformer.to(device)
print(f"✅ Model ready on device: {device}")

# ------------------------------------------------------------
# Prediction Endpoint
# ------------------------------------------------------------
@app.post("/predict")
def predict_fake_news(data: NewsInput):
    text = data.text

    # ----- Baseline Predictions -----
    Xv = tfidf.transform([text])
    baseline_results = {
        "LogisticRegression": int(lr.predict(Xv)[0]),
        "NaiveBayes": int(nb.predict(Xv)[0]),
        "SVM": int(svm.predict(Xv)[0])
    }

    # ----- Transformer Prediction -----
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = transformer(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        label = int(np.argmax(probs))
        confidence = float(np.max(probs))

    transformer_result = {
        "prediction": label,
        "confidence": round(confidence, 3)
    }

    # Convert numeric labels to readable form
    label_map = {0: "REAL", 1: "FAKE"}
    readable_baselines = {k: label_map[v] for k, v in baseline_results.items()}
    transformer_result["label"] = label_map[label]

    # Final decision (majority voting)
    votes = [v for v in baseline_results.values()] + [label]
    final_label = int(round(sum(votes) / len(votes)))
    final_label_readable = label_map[final_label]

    return {
        "Input_News": text,
        "Baseline_Models": readable_baselines,
        "Transformer": transformer_result,
        "Final_Decision": final_label_readable
    }

# ------------------------------------------------------------
# Root Endpoint
# ------------------------------------------------------------
@app.get("/")
def home():
    return {"message": "Welcome to the Fake News Detection API. Go to /docs for interactive testing."}

# ------------------------------------------------------------
# Run API (for local debugging)
# ------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=True)
