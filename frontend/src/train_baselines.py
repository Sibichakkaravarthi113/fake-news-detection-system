import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# Load train/test data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

X_train, y_train = train["content"], train["label"]
X_test, y_test = test["content"], test["label"]

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
Xv_train = tfidf.fit_transform(X_train)
Xv_test = tfidf.transform(X_test)

# Models
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "NaiveBayes": MultinomialNB(),
    "SVM": LinearSVC()
}

results = {}

for name, model in models.items():
    print(f"🔹 Training {name}...")
    model.fit(Xv_train, y_train)
    preds = model.predict(Xv_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))

    # Save model
    joblib.dump(model, f"models/{name}.joblib")

# Save vectorizer
joblib.dump(tfidf, "models/tfidf.joblib")

print("✅ All baseline models trained and saved in models/")
print("📊 Results:", results)
