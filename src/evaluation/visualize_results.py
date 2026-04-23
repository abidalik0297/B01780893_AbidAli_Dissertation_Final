import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

# ================================
# PATHS
# ================================

TEST_FILE = r"D:\abid_uws\data\processed\test_dataset.csv"

TFIDF_MODEL = r"D:\abid_uws\outputs\models\tfidf_logistic_model.pkl"
BERT_MODEL = r"D:\abid_uws\outputs\models\distilbert_phishing"
OCSVM_MODEL = r"D:\abid_uws\outputs\models\oneclass_svm.pkl"

EMBEDDINGS_FILE = r"D:\abid_uws\data\processed\sbert_embeddings.npy"

OUTPUT_DIR = r"D:\abid_uws\outputs\figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================================
# LOAD DATA
# ================================

df = pd.read_csv(TEST_FILE)

X = df["text"]
y = df["label"]


# ================================
# 1. TF-IDF MODEL
# ================================

print("Loading TF-IDF model...")

tfidf_model = joblib.load(TFIDF_MODEL)

tfidf_preds = tfidf_model.predict(X)
tfidf_probs = tfidf_model.predict_proba(X)[:, 1]

cm = confusion_matrix(y, tfidf_preds)

disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.title("TF-IDF Confusion Matrix")
plt.savefig(f"{OUTPUT_DIR}/tfidf_cm.png")
plt.close()


# ================================
# 2. DISTILBERT MODEL
# ================================

print("Loading DistilBERT model...")

tokenizer = DistilBertTokenizerFast.from_pretrained(BERT_MODEL)
model = DistilBertForSequenceClassification.from_pretrained(BERT_MODEL)

bert_preds = []
bert_probs = []

for text in X.tolist():
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=96)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)

    bert_preds.append(torch.argmax(probs).item())
    bert_probs.append(probs[0][1].item())

cm = confusion_matrix(y, bert_preds)

disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.title("DistilBERT Confusion Matrix")
plt.savefig(f"{OUTPUT_DIR}/bert_cm.png")
plt.close()


# ================================
# 3. ONE-CLASS SVM
# ================================

print("Loading OCSVM model...")

ocsvm = joblib.load(OCSVM_MODEL)

X_emb = np.load(EMBEDDINGS_FILE)

preds = ocsvm.predict(X_emb)
preds = np.array([0 if p == 1 else 1 for p in preds])

cm = confusion_matrix(y, preds)

disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.title("OC-SVM Confusion Matrix")
plt.savefig(f"{OUTPUT_DIR}/ocsvm_cm.png")
plt.close()


# ================================
# ROC CURVES
# ================================

fpr1, tpr1, _ = roc_curve(y, tfidf_probs)
fpr2, tpr2, _ = roc_curve(y, bert_probs)

plt.plot(fpr1, tpr1, label="TF-IDF")
plt.plot(fpr2, tpr2, label="DistilBERT")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.savefig(f"{OUTPUT_DIR}/roc_curve.png")
plt.close()


# ================================
# MODEL COMPARISON BAR CHART
# ================================

models = ["TF-IDF", "DistilBERT", "OC-SVM"]
f1_scores = [0.72, 0.86, 0.50]  # your results

plt.bar(models, f1_scores)
plt.title("Model Comparison (F1 Score)")
plt.ylabel("F1 Score")

plt.savefig(f"{OUTPUT_DIR}/model_comparison.png")
plt.close()

print("\nAll visualizations saved in:", OUTPUT_DIR)