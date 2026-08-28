import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# -----------------------------
# Text cleaning
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("labeled_data.csv")
df = df[["tweet", "class"]]
df["tweet"] = df["tweet"].apply(clean_text)

# -----------------------------
# Split data
# -----------------------------
X = df["tweet"]
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Vectorization (matches app1.py)
# -----------------------------
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english",
    ngram_range=(1, 2)
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -----------------------------
# Train ML model (matches app1.py)
# -----------------------------
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train_vec, y_train)

# -----------------------------
# Evaluation
# -----------------------------
y_pred = model.predict(X_test_vec)

report = classification_report(
    y_test, y_pred, target_names=["Hate Speech", "Offensive", "Normal"]
)
print("=== Classification Report ===")
print(report)

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred)

ConfusionMatrixDisplay(cm, display_labels=["Hate Speech", "Offensive", "Normal"]).plot()
plt.title("ML Confusion Matrix")
plt.savefig("ml_confusion_matrix.png")
plt.close()

# -----------------------------
# Save results
# -----------------------------
with open("model_summary.txt", "w") as f:
    f.write("=== Classification Report ===\n")
    f.write(report)

results_df = pd.DataFrame({
    "Tweet": X_test,
    "TrueLabel": y_test,
    "PredictedLabel": y_pred
})
results_df.to_csv("ml_results.csv", index=False)

print("Evaluation complete. Files generated: ml_confusion_matrix.png, model_summary.txt, ml_results.csv")