import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

# Load predictions
df = pd.read_csv("results/evaluation_reports/test_predictions.csv")
y_true = df['label_id'].values
y_probs = df['xlm_prob_fake'].values  # transformer probabilities
y_pred = df['xlm_pred'].values

# ROC curve
fpr, tpr, _ = roc_curve(y_true, y_probs)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1],[0,1],'--',color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - XLM-RoBERTa")
plt.legend()
plt.savefig("results/evaluation_reports/roc_curve.png")
plt.close()

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real","Fake"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - XLM-RoBERTa")
plt.savefig("results/evaluation_reports/confusion_matrix.png")
plt.close()

print("✅ Saved roc_curve.png and confusion_matrix.png in results/evaluation_reports/")
