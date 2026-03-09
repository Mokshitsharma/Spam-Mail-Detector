import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# -------- Configuration --------
CSV_PATH = os.environ.get("SPAM_CSV", "spam.csv")  
TEXT_COL = "text"
LABEL_COL = "label"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# -------- Load data --------
def load_dataset():
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH, encoding="utf-8")
        # Try to auto-detect columns if needed
        possible_label_cols = [c for c in df.columns if "label" in c.lower() or c.lower() in ("category","class")]
        possible_text_cols  = [c for c in df.columns if "text" in c.lower() or "message" in c.lower() or "sms" in c.lower()]
        label = possible_label_cols[0] if possible_label_cols else LABEL_COL
        text  = possible_text_cols[0] if possible_text_cols else TEXT_COL
        df = df.rename(columns={label: LABEL_COL, text: TEXT_COL})
    else:
        # tiny fallback sample (so the script always runs)
        df = pd.DataFrame({
            LABEL_COL: ["ham","ham","spam","ham","spam","spam","ham","spam"],
            TEXT_COL: [
                "hey are we still meeting today?",
                "please call me when you reach",
                "WINNER!! click this link to claim your prize",
                "see you at the office",
                "free entry in 2 a weekly competition!!!",
                "URGENT! you have won a car",
                "can you send the file?",
                "claim your FREE coupons now"
            ]
        })
    # Normalize labels
    df[LABEL_COL] = df[LABEL_COL].str.lower().str.strip().map({"spam":"spam", "ham":"ham"}).fillna(df[LABEL_COL])
    return df[[TEXT_COL, LABEL_COL]].dropna()

df = load_dataset()
print(f"Loaded {len(df)} messages")

# -------- Split --------
X_train, X_test, y_train, y_test = train_test_split(
    df[TEXT_COL], df[LABEL_COL], test_size=0.2, random_state=42, stratify=df[LABEL_COL] if df[LABEL_COL].nunique()==2 else None
)

# -------- Vectorize --------
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1,2),           # unigrams + bigrams often help
    min_df=2                      # ignore extremely rare terms (tweak for small datasets)
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

# -------- Train baseline: MultinomialNB --------
nb = MultinomialNB()
nb.fit(X_train_vec, y_train)
nb_preds = nb.predict(X_test_vec)
print("\n=== MultinomialNB ===")
print("Accuracy:", accuracy_score(y_test, nb_preds))
print(classification_report(y_test, nb_preds))
print("Confusion matrix:\n", confusion_matrix(y_test, nb_preds))

# -------- Optional second model: Linear SVM --------
svm = LinearSVC()
svm.fit(X_train_vec, y_train)
svm_preds = svm.predict(X_test_vec)
print("\n=== LinearSVC ===")
print("Accuracy:", accuracy_score(y_test, svm_preds))
print(classification_report(y_test, svm_preds))
print("Confusion matrix:\n", confusion_matrix(y_test, svm_preds))

# -------- Persist the best model (choose by accuracy) --------
best_model, best_name, best_acc = (nb, "MultinomialNB", accuracy_score(y_test, nb_preds))
svm_acc = accuracy_score(y_test, svm_preds)
if svm_acc >= best_acc:
    best_model, best_name, best_acc = (svm, "LinearSVC", svm_acc)

joblib.dump(best_model, f"{MODEL_DIR}/spam_model.joblib")
joblib.dump(vectorizer, f"{MODEL_DIR}/tfidf.joblib")
print(f"\nSaved best model: {best_name} (acc={best_acc:.3f}) to {MODEL_DIR}/")

# -------- Inference helper --------
def predict_message(msg: str):
    vec = vectorizer.transform([msg])
    return best_model.predict(vec)[0]

# Demo
samples = [
    "Claim your FREE prize by clicking this link!",
    "Can we reschedule our meeting to 5 pm?"
]
for s in samples:
    print(f"Text: {s}\n -> {predict_message(s)}\n")
