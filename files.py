import os
import pandas as pd

# ============================================
# CONFIGURATION
# ============================================

ENRON_ROOT = r"D:\abid_uws\data\raw\enron_mail_20150507\maildir"

OUTPUT_FILE = r"D:\abid_uws\data\processed\enron_file_index.csv"


# ============================================
# INDEX ALL EMAIL FILES
# ============================================

def index_enron_dataset(root_path):

    email_paths = []

    for root, dirs, files in os.walk(root_path):

        for file in files:

            full_path = os.path.join(root, file)

            # ensure it's actually a file
            if os.path.isfile(full_path):
                email_paths.append(full_path)

    return email_paths


# ============================================
# MAIN
# ============================================

def main():

    print("Scanning Enron maildir dataset...")
    print("Root folder:", ENRON_ROOT)

    email_files = index_enron_dataset(ENRON_ROOT)

    print("Total email files found:", len(email_files))

    df = pd.DataFrame(email_files, columns=["file_path"])

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nIndex saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()