import numpy as np
import joblib
from sklearn.metrics import classification_report

# ======================================================
# PATHS
# ======================================================

EMBEDDINGS_FILE = r"D:\abid_uws\data\processed\sbert_embeddings.npy"
LABEL_FILE = r"D:\abid_uws\data\processed\sbert_labels.npy"

MODEL_PATH = r"D:\abid_uws\outputs\models\oneclass_svm.pkl"

# ======================================================
# LOAD
# ======================================================

X = np.load(EMBEDDINGS_FILE)
y = np.load(LABEL_FILE)

model = joblib.load(MODEL_PATH)

# ======================================================
# PREDICT
# ======================================================

print("Evaluating anomaly detection...\n")

preds = model.predict(X)

# Convert:
# +1 → legitimate (0)
# -1 → anomaly (1)

preds = [0 if p == 1 else 1 for p in preds]

print(classification_report(y, preds))