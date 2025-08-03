def chunk_text(texts, chunk_size=100):
    chunks = []
    for doc_id, text in enumerate(texts):
        words = text.split()
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append({
                "id": f"doc{doc_id}_chunk{i // chunk_size}",
                "text": chunk
            })
    return chunks
