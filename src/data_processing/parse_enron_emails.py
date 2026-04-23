import pandas as pd
import os
import re
from tqdm import tqdm


INDEX_FILE = r"D:\abid_uws\data\processed\enron_file_index.csv"
OUTPUT_FILE = r"D:\abid_uws\data\processed\enron_legitimate_emails.csv"


def extract_email_body(text):

    parts = text.split("\n\n", 1)

    if len(parts) > 1:
        return parts[1]
    else:
        return text


def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)

    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def parse_emails():

    df = pd.read_csv(INDEX_FILE)

    email_texts = []

    for path in tqdm(df["file_path"], desc="Parsing emails"):

        try:

            with open(path, "r", encoding="latin1") as f:

                raw = f.read()

                body = extract_email_body(raw)

                cleaned = clean_text(body)

                if len(cleaned) > 20:
                    email_texts.append(cleaned)

        except:
            continue

    dataset = pd.DataFrame({
        "text": email_texts,
        "label": 0
    })

    dataset.to_csv(OUTPUT_FILE, index=False)

    print("\nSaved cleaned Enron dataset to:")
    print(OUTPUT_FILE)

    print("\nTotal legitimate emails:", len(dataset))


if __name__ == "__main__":
    parse_emails()