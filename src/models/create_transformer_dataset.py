import pandas as pd
from sklearn.model_selection import train_test_split

LEGIT_FILE = r"D:\abid_uws\data\processed\enron_legitimate_emails.csv"
PHISH_FILE = r"D:\abid_uws\data\processed\phishing_emails.csv"

OUTPUT_DIR = r"D:\abid_uws\data\processed"


def build_transformer_dataset():

    print("Loading datasets...")

    legit = pd.read_csv(LEGIT_FILE)
    phish = pd.read_csv(PHISH_FILE)

    print("Original legitimate:", len(legit))
    print("Original phishing:", len(phish))

    # ------------------------------------------------
    # sample balanced dataset
    # ------------------------------------------------

    legit_sample = legit.sample(n=150000, random_state=42)
    phish_sample = phish.sample(n=150000, random_state=42)

    dataset = pd.concat([legit_sample, phish_sample])

    dataset = dataset.sample(frac=1, random_state=42)

    print("Final dataset size:", len(dataset))

    train, test = train_test_split(
        dataset,
        test_size=0.1,
        stratify=dataset["label"],
        random_state=42
    )

    train.to_csv(f"{OUTPUT_DIR}/bert_train.csv", index=False)
    test.to_csv(f"{OUTPUT_DIR}/bert_test.csv", index=False)

    print("\nSaved transformer datasets")
    print("Train:", len(train))
    print("Test:", len(test))


if __name__ == "__main__":
    build_transformer_dataset()