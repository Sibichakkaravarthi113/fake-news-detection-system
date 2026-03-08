# src/evaluate_models.py
import joblib, pandas as pd, numpy as np
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizerFast
import torch, torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# === Load test data ===
test = pd.read_csv("data/processed/test.csv")   # make sure test.csv exists
if 'label' in test.columns and test['label'].dtype == 'O':
    test['label_id'] = test['label'].map({'real':0, 'fake':1})
else:
    test['label_id'] = test['label']

# === Load models ===
tfidf = joblib.load("models/tfidf.joblib")
lr = joblib.load("models/LogisticRegression.joblib")

tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")
model = XLMRobertaForSequenceClassification.from_pretrained("models/xlm_model")
model.eval()
if torch.cuda.is_available():
    model.to("cuda")

# === Logistic Regression predictions ===
X_test = tfidf.transform(test['text'].tolist())
lr_probs = lr.predict_proba(X_test)[:,1]
lr_preds = (lr_probs >= 0.5).astype(int)

# === Transformer predictions (batched) ===
def predict_xlm_texts(texts, batch=16):
    probs = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i+batch]
        enc = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=256)
        if torch.cuda.is_available():
            enc = {k:v.to("cuda") for k,v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            p = F.softmax(logits, dim=-1).cpu().numpy()[:,1]  # probability of "fake"
        probs.extend(p.tolist())
    return np.array(probs)

xlm_probs = predict_xlm_texts(test['text'].tolist(), batch=8)
xlm_preds = (xlm_probs >= 0.5).astype(int)

# === Reports ===
def report(y_true, y_pred, probs=None, name="Model"):
    print("=== ", name)
    print(classification_report(y_true, y_pred, target_names=["real","fake"], zero_division=0))
    if probs is not None and len(np.unique(y_true)) > 1:
        try:
            print("ROC-AUC:", roc_auc_score(y_true, probs))
        except Exception as e:
            print("ROC-AUC error:", e)
    print("Confusion:\n", confusion_matrix(y_true, y_pred))

y_true = test["label_id"].values
report(y_true, lr_preds, probs=lr_probs, name="Logistic Regression")
report(y_true, xlm_preds, probs=xlm_probs, name="XLM-RoBERTa")

# === Save predictions ===
out = test.copy()
out["lr_prob_fake"] = lr_probs
out["lr_pred"] = lr_preds
out["xlm_prob_fake"] = xlm_probs   # ✅ this column will be used by plots.py
out["xlm_pred"] = xlm_preds

import os
os.makedirs("results/evaluation_reports", exist_ok=True)
out.to_csv("results/evaluation_reports/test_predictions.csv", index=False)

print("✅ Saved test_predictions.csv in results/evaluation_reports/")
