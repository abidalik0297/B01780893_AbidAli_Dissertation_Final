import pandas as pd
import os
import json
import joblib
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline


# ==========================================================
# PATH CONFIGURATION
# ==========================================================

TRAIN_FILE = r"D:\abid_uws\data\processed\train_dataset.csv"
TEST_FILE = r"D:\abid_uws\data\processed\test_dataset.csv"

MODEL_DIR = r"D:\abid_uws\outputs\models"

MODEL_PATH = os.path.join(MODEL_DIR, "tfidf_logistic_model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "tfidf_logistic_metrics.json")


# ==========================================================
# TRAINING FUNCTION
# ==========================================================

def train_model():

    start_time = time.time()

    print("\n==============================")
    print("PHISHING DETECTION BASELINE")
    print("TF-IDF + LOGISTIC REGRESSION")
    print("==============================\n")

    print("Loading datasets...\n")

    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)

    X_train = train["text"]
    y_train = train["label"]

    X_test = test["text"]
    y_test = test["label"]

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))


    # ======================================================
    # MODEL PIPELINE
    # ======================================================

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50000,
            ngram_range=(1,2),
            stop_words="english"
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            n_jobs=-1,
            class_weight="balanced"
        ))
    ])

    print("\nTraining model...\n")

    model.fit(X_train, y_train)

    print("Training complete.")


    # ======================================================
    # EVALUATION
    # ======================================================

    print("\nEvaluating model...\n")

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, predictions, output_dict=True)

    print(classification_report(y_test, predictions))

    auc = roc_auc_score(y_test, probabilities)

    print("ROC-AUC:", auc)


    # ======================================================
    # SAVE ARTIFACTS
    # ======================================================

    os.makedirs(MODEL_DIR, exist_ok=True)

    # save trained pipeline
    joblib.dump(model, MODEL_PATH)

    print("\nModel saved to:")
    print(MODEL_PATH)


    # save metrics
    metrics = {
        "roc_auc": auc,
        "accuracy": report["accuracy"],
        "precision_phishing": report["1"]["precision"],
        "recall_phishing": report["1"]["recall"],
        "f1_phishing": report["1"]["f1-score"]
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    print("\nMetrics saved to:")
    print(METRICS_PATH)


    # ======================================================
    # TRAINING TIME
    # ======================================================

    end_time = time.time()
    duration = (end_time - start_time) / 60

    print("\nTraining time: %.2f minutes" % duration)


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    train_model()