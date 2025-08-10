import os
import pdfplumber
import re
from pathlib import Path

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
Path(PROCESSED_DIR).mkdir(parents=True, exist_ok=True)

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + ""
    return text

def clean_text(text):
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\d+ of \d+', '', text)  # remove page numbers
    text = re.sub(r'\s{2,}', ' ', text)
    return text

for fname in os.listdir(RAW_DIR):
    if fname.endswith(".pdf"):
        raw_text = extract_text_from_pdf(os.path.join(RAW_DIR, fname))
        clean = clean_text(raw_text)
        with open(os.path.join(PROCESSED_DIR, fname.replace(".pdf", ".txt")), "w") as f:
            f.write(clean)