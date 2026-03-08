import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# load predictions saved from evaluate_models.py
df = pd.read_csv("results/evaluation_reports/test_predictions.csv")
y_true = df['label_id'].values

# logistic regression
lr_preds = df['lr_pred'].values
# transformer
xlm_preds = df['xlm_pred'].values

def metrics_table(y_true, preds, name):
    acc = accuracy_score(y_true, preds)
    p, r, f1, _ = precision_recall_fscore_support(y_true, preds, average='binary', pos_label=1, zero_division=0)
    return {"model":name, "accuracy":acc, "precision":p, "recall":r, "f1":f1}

rows = []
rows.append(metrics_table(y_true, lr_preds, "Logistic Regression"))
rows.append(metrics_table(y_true, xlm_preds, "XLM-RoBERTa"))

out = pd.DataFrame(rows)
print(out)
out.to_csv("results/evaluation_reports/metrics_summary.csv", index=False)
