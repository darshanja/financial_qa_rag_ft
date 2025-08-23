import os
import pdfplumber
import re
from pathlib import Path
from utils.chunking import smart_chunk_text

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
CHUNKS_DIR = "data/chunks"

Path(CHUNKS_DIR).mkdir(parents=True, exist_ok=True)
Path(PROCESSED_DIR).mkdir(parents=True, exist_ok=True)

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:  # skip empty pages
                text += page_text + "\n"
    return text


def clean_text(text: str) -> str:
    # Remove common headers/footers
    text = re.sub(r'Allstate.*?\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Page \d+ of \d+', '', text)

    # Fix broken numbers: "57 , 094" → "57,094"
    text = re.sub(r'(\d)\s*,\s*(\d)', r'\1,\2', text)

    # Fix broken words like "T o t a l" → "Total" (only when letters are isolated)
    text = re.sub(r'(?<=\b\w) (?=\w\b)', '', text)

    # Normalize spaces/newlines
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', '\n', text)

    # Remove stray lines: pure digits, year-only, or too short
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if len(line) <= 5:
            continue
        if re.fullmatch(r"\d{4}", line):  # year like 2023
            continue
        if re.fullmatch(r"[\d,\. ]+", line):  # only numbers
            continue
        lines.append(line)

    return "\n".join(lines).strip()


# Process all PDFs
for fname in os.listdir(RAW_DIR):
    if fname.endswith(".pdf"):
        raw_text = extract_text_from_pdf(os.path.join(RAW_DIR, fname))
        clean = clean_text(raw_text)

        # Save cleaned text
        with open(os.path.join(PROCESSED_DIR, fname.replace(".pdf", ".txt")), "w", encoding="utf-8") as f:
            f.write(clean)

        # Chunk and save
        chunks = smart_chunk_text([clean], chunk_size=300, overlap=50)
        with open(os.path.join(CHUNKS_DIR, fname.replace(".pdf", "_chunks.txt")), "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(chunk + "\n---\n")