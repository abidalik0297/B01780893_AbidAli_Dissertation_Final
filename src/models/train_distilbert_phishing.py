import os
import json
import time
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer
)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


# =====================================================
# PATHS
# =====================================================

TRAIN_FILE = r"D:\abid_uws\data\processed\bert_train.csv"
TEST_FILE = r"D:\abid_uws\data\processed\bert_test.csv"

OUTPUT_DIR = r"D:\abid_uws\outputs\models\distilbert_phishing"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================
# LOAD DATA
# =====================================================

print("\nLoading datasets...\n")

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print("Train samples:", len(train_df))
print("Test samples:", len(test_df))

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)


# =====================================================
# TOKENIZER
# =====================================================

print("\nLoading tokenizer...\n")

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")


def tokenize(example):

    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=96
    )


print("\nTokenizing datasets...\n")

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])


# =====================================================
# MODEL
# =====================================================

print("\nLoading DistilBERT model...\n")

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)


# =====================================================
# METRICS
# =====================================================

def compute_metrics(pred):

    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )

    acc = accuracy_score(labels, preds)

    probs = torch.softmax(torch.tensor(pred.predictions), dim=1)[:, 1]

    auc = roc_auc_score(labels, probs)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": auc
    }


# =====================================================
# TRAINING ARGUMENTS (OLD API COMPATIBLE)
# =====================================================

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    learning_rate=2e-5,

    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    num_train_epochs=2,

    weight_decay=0.01,

    logging_steps=500,

    save_strategy="no",   # <-- disables mid-training checkpoints

    do_train=True,
    do_eval=True
)


# =====================================================
# TRAINER
# =====================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)


# =====================================================
# TRAIN MODEL
# =====================================================

print("\nStarting training...\n")

start_time = time.time()

trainer.train()

end_time = time.time()

print("\nTraining finished.")

print("Training time (minutes):", (end_time - start_time) / 60)


# =====================================================
# EVALUATION
# =====================================================

print("\nEvaluating model...\n")

metrics = trainer.evaluate()

print(metrics)


# =====================================================
# SAVE MODEL
# =====================================================

print("\nSaving model...\n")

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")

with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)

print("\nModel saved to:", OUTPUT_DIR)
print("Metrics saved to:", metrics_path)