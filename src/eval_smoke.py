# src/eval_smoke.py
import joblib
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizerFast
import torch, torch.nn.functional as F

# load baseline LR & tfidf
tfidf = joblib.load("models/tfidf.joblib")
lr = joblib.load("models/LogisticRegression.joblib")

# load transformer
tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")
model = XLMRobertaForSequenceClassification.from_pretrained("models/xlm_model")
model.eval()
if torch.cuda.is_available(): model.to('cuda')

def predict_transformer(text):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    if torch.cuda.is_available():
        enc = {k:v.to('cuda') for k,v in enc.items()}
    with torch.no_grad():
        out = model(**enc).logits
        probs = F.softmax(out, dim=-1).cpu().numpy()[0]
    # assume label 0=real, 1=fake
    return {"prob_fake": float(probs[1]), "prob_real": float(probs[0])}

def predict_lr(text):
    x = tfidf.transform([text])
    proba = lr.predict_proba(x)[0]    # [prob_real, prob_fake]
    return {"prob_fake": float(proba[1]), "prob_real": float(proba[0])}

text = "MrBeast has 5 million subscribers"
print("LR:", predict_lr(text))
print("XLM:", predict_transformer(text))
