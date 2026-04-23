import pandas as pd
from sklearn.model_selection import train_test_split
import os

LEGIT_FILE = r"D:\abid_uws\data\processed\enron_legitimate_emails.csv"
PHISH_FILE = r"D:\abid_uws\data\processed\phishing_emails.csv"

OUTPUT_DIR = r"D:\abid_uws\data\processed"


def build_dataset():

    print("Loading datasets...")

    legit = pd.read_csv(LEGIT_FILE)
    phish = pd.read_csv(PHISH_FILE)

    print("Legitimate emails:", len(legit))
    print("Phishing emails:", len(phish))

    # ------------------------------------------------
    # create zero-day phishing subset (20%)
    # ------------------------------------------------

    phish_train, phish_zero = train_test_split(
        phish,
        test_size=0.2,
        random_state=42
    )

    # ------------------------------------------------
    # combine legitimate + phishing for training
    # ------------------------------------------------

    dataset = pd.concat([legit, phish_train], ignore_index=True)

    train, test = train_test_split(
        dataset,
        test_size=0.2,
        stratify=dataset["label"],
        random_state=42
    )

    # ------------------------------------------------
    # build zero-day evaluation dataset
    # ------------------------------------------------

    zero_day = pd.concat([test[test["label"] == 0], phish_zero])

    # ------------------------------------------------
    # save datasets
    # ------------------------------------------------

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train.to_csv(os.path.join(OUTPUT_DIR, "train_dataset.csv"), index=False)
    test.to_csv(os.path.join(OUTPUT_DIR, "test_dataset.csv"), index=False)
    zero_day.to_csv(os.path.join(OUTPUT_DIR, "zero_day_dataset.csv"), index=False)

    print("\nDatasets saved:")
    print("Train size:", len(train))
    print("Test size:", len(test))
    print("Zero-day test size:", len(zero_day))


if __name__ == "__main__":
    build_dataset()