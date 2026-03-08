# ============================================================
#  Fake News Detection API - Baselines + Transformer + Fact Checker (Hybrid)
# ============================================================
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0  # for consistent results
from src.fact_check_google import fact_check as run_fact_check
from fastapi import FastAPI
from pydantic import BaseModel
import joblib, torch
from transformers import XLMRobertaTokenizerFast, XLMRobertaForSequenceClassification
import numpy as np
import uvicorn
# import fact_check_aws  # ✅ add this import

# ------------------------------------------------------------
# Initialize FastAPI App
# ------------------------------------------------------------
app = FastAPI(
    title="Fake News Detection API (Hybrid Model)",
    description="Combines baseline ML models, XLM-RoBERTa transformer, and AWS-based fact checking.",
    version="3.0"
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
        # ----- Language Detection -----
    try:
        lang_code = detect(text)
    except Exception:
        lang_code = "unknown"

    # Optional: simple mapping for display
    lang_map = {
        "en": "English",
        "hi": "Hindi",
        "ta": "Tamil",
        "te": "Telugu",
        "ml": "Malayalam",
        "kn": "Kannada",
        "fr": "French",
        "es": "Spanish",
    }
    lang_name = lang_map.get(lang_code, "Unknown")


    # ----- Baseline Predictions -----
    baselines = {}
    for name, model in {
    	"Logistic Regression": lr,
    	"Naive Bayes": nb,
    	"SVM": svm
    }.items():
        try:
            text_vec = tfidf.transform([text])
        
            if hasattr(model, "predict_proba"):
                # Models with probability support
                proba = model.predict_proba(text_vec)[0]
                confidence = round(max(proba) * 100, 2)
                label = "FAKE" if np.argmax(proba) == 1 else "REAL"
        
            elif hasattr(model, "decision_function"):
                # For SVM or similar models
                score = model.decision_function(text_vec)[0]
                confidence = round((abs(score) / (1 + abs(score))) * 100, 2)  # normalized
                label = "FAKE" if score > 0 else "REAL"
        
            else:
                # Fallback case
                label = model.predict(text_vec)[0]
                confidence = 50.0

        except Exception as e:
            print(f"[Error in {name}] {e}")
            label = "UNKNOWN"
            confidence = 0.0

        baselines[name] = {"prediction": label, "confidence": confidence}



    # ----- Transformer Prediction -----
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = transformer(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        transformer_label = "FAKE" if np.argmax(probs) == 1 else "REAL"
        transformer_conf = round(float(np.max(probs)) * 100, 2)

    # ----- Fact Checker -----
     # ----- Fact Checker (Google via SerpAPI) -----
    try:
        fact_result = run_fact_check(text)
        fact_verdict = fact_result.get("verdict", "UNVERIFIED")   # REAL / UNVERIFIED
    except Exception as e:
        fact_result = {"error": str(e)}
        fact_verdict = "UNVERIFIED"

    # ----- Final Verdict Logic (Hybrid Decision) -----
    if transformer_label == "REAL" and fact_verdict == "REAL":
        final_verdict = "REAL"
        reason = "Both transformer and Google-based fact-checker agree the news is real."
    elif transformer_label == "FAKE" and fact_verdict == "REAL":
        final_verdict = "CONFLICT"
        reason = "Transformer marked as fake, but Google search found similar real news on trusted sources."
    elif transformer_label == "FAKE" and fact_verdict == "UNVERIFIED":
        final_verdict = "FAKE"
        reason = "Transformer marked as fake and no strong evidence was found on trusted news sites."
    elif transformer_label == "REAL" and fact_verdict == "UNVERIFIED":
        final_verdict = "UNVERIFIED"
        reason = "Transformer says real, but Google did not return strong confirmation."
    else:
        final_verdict = "UNVERIFIED"
        reason = "Insufficient data for a confident decision."

    return {
        "input_text": text,
        "language": {
            "code": lang_code,
            "name": lang_name
        },
        "baselines": baselines,
        "transformer": {
            "prediction": transformer_label,
            "confidence": transformer_conf
        },
        "fact_check": fact_result,
        "final_verdict": final_verdict,
        "reason": reason
    }

# ------------------------------------------------------------
# Root Endpoint
# ------------------------------------------------------------
@app.get("/")
def home():
    return {"message": "Welcome to the Fake News Detection API (Hybrid Model). Visit /docs to test."}

# ------------------------------------------------------------
# Run API (for local debugging)
# ------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=True)
