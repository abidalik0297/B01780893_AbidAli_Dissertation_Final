import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np

# ======================================================
# PATHS
# ======================================================

INPUT_FILE = r"D:\abid_uws\data\processed\train_dataset.csv"
OUTPUT_FILE = r"D:\abid_uws\data\processed\sbert_embeddings.npy"
LABEL_FILE = r"D:\abid_uws\data\processed\sbert_labels.npy"

# ======================================================
# LOAD MODEL
# ======================================================

print("Loading SBERT model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

# ======================================================
# LOAD DATA
# ======================================================

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print("Total samples:", len(df))

texts = df["text"].astype(str).tolist()
labels = df["label"].values

# ======================================================
# GENERATE EMBEDDINGS
# ======================================================

print("\nGenerating embeddings (FAST BATCH MODE)...\n")

embeddings = model.encode(
    texts,
    batch_size=64,          # you can try 32 if RAM issues
    show_progress_bar=True
)

# ======================================================
# SAVE
# ======================================================

np.save(OUTPUT_FILE, embeddings)
np.save(LABEL_FILE, labels)

print("\nEmbeddings saved to:", OUTPUT_FILE)
print("Labels saved to:", LABEL_FILE)