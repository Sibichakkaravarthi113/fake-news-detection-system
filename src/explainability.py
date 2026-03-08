import joblib
import torch
import pandas as pd
from lime.lime_text import LimeTextExplainer
from transformers import XLMRobertaTokenizerFast, XLMRobertaForSequenceClassification
from captum.attr import IntegratedGradients

# Load Logistic Regression + TFIDF
tfidf = joblib.load("models/tfidf.joblib")
lr = joblib.load("models/LogisticRegression.joblib")

# LIME for Logistic Regression
explainer = LimeTextExplainer(class_names=['real','fake'])

def predict_proba_lr(texts):
    return lr.predict_proba(tfidf.transform(texts))

text = "Miracle pill cured cancer overnight"
lime_exp = explainer.explain_instance(text, predict_proba_lr, num_features=6)

# Save LIME explanation
lime_out = pd.DataFrame(lime_exp.as_list(), columns=["word", "weight"])
lime_out.to_csv("results/explainability/lime_lr_example.csv", index=False)

# Captum for Transformer
tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")
model = XLMRobertaForSequenceClassification.from_pretrained("models/xlm_model")
model.eval()

ig = IntegratedGradients(lambda ids, mask: model(ids, attention_mask=mask).logits[:,1])

enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=256)
attr, _ = ig.attribute(inputs=enc['input_ids'], additional_forward_args=(enc['attention_mask'],), target=1, return_convergence_delta=True)

tokens = tokenizer.convert_ids_to_tokens(enc['input_ids'].squeeze().tolist())
scores = [float(a) for a in attr.squeeze().sum(dim=-1)]

# Save Captum explanation
df_captum = pd.DataFrame({"token":tokens, "importance":scores})
df_captum.to_csv("results/explainability/captum_xlm_example.csv", index=False)

print("✅ LIME + Captum explanations saved in results/explainability/")
