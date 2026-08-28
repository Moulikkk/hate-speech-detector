import pandas as pd
import re
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# -----------------------------
# Text preprocessing function
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)   # remove URLs
    text = re.sub(r"[^a-z\s]", "", text)         # remove punctuation & numbers
    text = re.sub(r"\s+", " ", text)             # remove extra spaces
    return text.strip()

# -----------------------------
# Load dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("labeled_data.csv")
    df = df[["tweet", "class"]]
    df["tweet"] = df["tweet"].apply(clean_text)
    return df

df = load_data()

# -----------------------------
# Split data
# -----------------------------
X = df["tweet"]
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Vectorization (TF-IDF)
# -----------------------------
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english",
    ngram_range=(1, 2)
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -----------------------------
# Train ML model
# -----------------------------
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train_vec, y_train)

# -----------------------------
# Evaluate model (printed to terminal, not the UI)
# -----------------------------
y_pred = model.predict(X_test_vec)
print("=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=["Hate Speech", "Offensive", "Normal"]))

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Hate Speech Detection (ML Based)")
st.write("This app detects **Hate Speech**, **Offensive Language**, or **Normal Speech**.")

user_input = st.text_area("Enter a sentence:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned_input = clean_text(user_input)
        input_vec = vectorizer.transform([cleaned_input])
        prediction = model.predict(input_vec)[0]

        if prediction == 0:
            st.error("❌ Hate Speech ")
        elif prediction == 1:
            st.warning("⚠️ Offensive Language ")
        else:
            st.success("✅ Normal / Safe Speech ")