import numpy as np
import joblib
from sklearn.svm import OneClassSVM

# ======================================================
# PATHS
# ======================================================

EMBEDDINGS_FILE = r"D:\abid_uws\data\processed\sbert_embeddings.npy"
LABEL_FILE = r"D:\abid_uws\data\processed\sbert_labels.npy"

MODEL_PATH = r"D:\abid_uws\outputs\models\oneclass_svm.pkl"

# ======================================================
# LOAD DATA
# ======================================================

print("Loading embeddings...")

X = np.load(EMBEDDINGS_FILE)
y = np.load(LABEL_FILE)

print("Total samples:", len(X))

# ======================================================
# TRAIN ONLY ON LEGITIMATE EMAILS
# ======================================================

X_legit = X[y == 0]

print("Training on legitimate emails only:", len(X_legit))

# ======================================================
# TRAIN MODEL
# ======================================================

model = OneClassSVM(
    kernel="rbf",
    gamma="scale",
    nu=0.15
)

print("\nTraining One-Class SVM...\n")

model.fit(X_legit)

# ======================================================
# SAVE MODEL
# ======================================================

joblib.dump(model, MODEL_PATH)

print("\nModel saved to:", MODEL_PATH)