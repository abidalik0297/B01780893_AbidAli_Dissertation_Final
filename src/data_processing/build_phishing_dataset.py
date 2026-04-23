import pandas as pd
import os
import re

PHISHING_FOLDER = r"D:\abid_uws\data\raw\phishing emails"
OUTPUT_FILE = r"D:\abid_uws\data\processed\phishing_emails.csv"


def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def load_phishing_files():

    datasets = []

    files = os.listdir(PHISHING_FOLDER)

    for file in files:

        if file.endswith(".csv"):

            path = os.path.join(PHISHING_FOLDER, file)

            print("Loading:", file)

            try:
                df = pd.read_csv(
                    path,
                    engine="python",
                    encoding="latin1",
                    on_bad_lines="skip"
                )
            except Exception as e:
                print("Skipping file due to error:", file)
                print(e)
                continue

            # find likely text column
            text_col = None

            for col in df.columns:
                if "text" in col.lower() or "body" in col.lower() or "email" in col.lower():
                    text_col = col
                    break

            if text_col is None:
                text_col = df.columns[0]

            df = df[[text_col]].rename(columns={text_col: "text"})

            df["text"] = df["text"].apply(clean_text)

            datasets.append(df)

    combined = pd.concat(datasets, ignore_index=True)

    combined["label"] = 1

    combined = combined[combined["text"].str.len() > 20]

    combined.to_csv(OUTPUT_FILE, index=False)

    print("\nSaved phishing dataset to:", OUTPUT_FILE)
    print("Total phishing emails:", len(combined))


if __name__ == "__main__":
    load_phishing_files()