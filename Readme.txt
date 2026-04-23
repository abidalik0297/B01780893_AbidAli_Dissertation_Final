# Phishing Email Detection System

## Setup Instructions

1. Create virtual environment:
python -m venv phishingproject

2. Activate environment:
phishingproject\Scripts\activate

3. Upgrade pip:
python -m pip install --upgrade pip

4. Install dependencies:
pip install -r requirements.txt

## Run the Project

streamlit run app.py

## Dataset

Due to size limitations, datasets and output files are not included in this repository.
They can be accessed from:
- Enron Email Dataset
- Nazario Phishing Corpus

## Notes

This project compares:
- TF-IDF with Logistic Regression
- DistilBERT
- One-Class SVM with SBERT embeddings
