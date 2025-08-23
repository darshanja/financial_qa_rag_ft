# utils/chunking.py
import re
import nltk
nltk.download("punkt", quiet=True)
from nltk.tokenize import sent_tokenize

def smart_chunk_text(text, chunk_size=300, overlap=50):
    # Ensure input is a string
    if isinstance(text, list):
        text = "\n".join(text)

    # Split text into paragraphs
    paragraphs = re.split(r"\n\s*\n", text)

    chunks = []
    for para in paragraphs:
        sentences = sent_tokenize(para)
        words = []
        for sent in sentences:
            sent_words = sent.split()
            # If sentence itself is longer than chunk_size, break it
            if len(sent_words) > chunk_size:
                for i in range(0, len(sent_words), chunk_size - overlap):
                    part = " ".join(sent_words[i:i+chunk_size])
                    if len(part.split()) > 30:
                        chunks.append(part)
            else:
                words.extend(sent_words)

            # If collected enough words, make a chunk
            if len(words) >= chunk_size:
                chunk = " ".join(words[:chunk_size])
                chunks.append(chunk)
                # Keep overlap
                words = words[chunk_size - overlap:]

        # Leftover words (end of paragraph)
        if words and len(words) > 30:
            chunks.append(" ".join(words))

    return chunks