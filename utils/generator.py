# utils/generator.py
from typing import List, Tuple
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import nltk, re
from nltk.tokenize import sent_tokenize
import torch
import functools

# Ensure punkt is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

# Model names
EXTRACTIVE_MODEL_NAME = "deepset/roberta-base-squad2"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# Load models once
device = 0 if torch.cuda.is_available() else -1
qa_pipeline = pipeline("question-answering", model=EXTRACTIVE_MODEL_NAME, device=device)
embedder = SentenceTransformer(
    EMBED_MODEL_NAME, device="cuda" if torch.cuda.is_available() else "cpu"
)


@functools.lru_cache(maxsize=512)
def embed_text(text: str):
    """Cache embeddings to avoid recomputation."""
    return embedder.encode(text, convert_to_tensor=True)


def _select_relevant_sentences(query: str, chunks: List[str], top_k: int = 3) -> str:
    """Select top-k most relevant sentences from retrieved chunks."""
    sentences = []
    for ch in chunks:
        sentences.extend(sent_tokenize(ch))

    # Filter out numeric/table junk
    sentences = [s for s in sentences if not re.fullmatch(r"[\d\W]+", s.strip())]

    if not sentences:
        return ""

    query_emb = embed_text(query)
    sent_embs = embedder.encode(sentences, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_emb, sent_embs)[0]
    top_results = cos_scores.topk(k=min(top_k, len(sentences)))
    selected = [sentences[idx] for idx in top_results[1]]
    return " ".join(selected)


def generate_answer(
    query: str,
    context_chunks: List[str],
) -> Tuple[str, str]:
    """
    Generate (answer, supporting_context) using extractive QA.
    """
    supporting_context = _select_relevant_sentences(query, context_chunks, top_k=5)

    if not supporting_context.strip():
        return ("I cannot find this information in the financial documents.", "")

    try:
        result = qa_pipeline({"question": query, "context": supporting_context})
        answer = normalize_answer(result.get("answer", "").strip())
        if not answer:
            return ("I cannot find this information in the financial documents.", supporting_context)

        refined_context = get_supporting_context(supporting_context, answer)
        return (answer, refined_context)

    except Exception as e:
        return (f"Error in extractive QA: {e}", supporting_context)


def normalize_answer(ans: str) -> str:
    """Normalize numeric answers like 57,094 -> $57.09 billion."""
    cleaned = ans.replace(",", "").replace("$", "").strip()
    if cleaned.isdigit():
        try:
            val = int(cleaned)
            if val >= 1e9:
                return f"${val/1e9:.2f} billion"
            elif val >= 1e6:
                return f"${val/1e6:.2f} million"
            else:
                return f"${val}"
        except Exception:
            return ans
    return ans


def get_supporting_context(context: str, answer: str, window: int = 1) -> str:
    """Return up to 2 sentences around the one containing the answer."""
    sentences = sent_tokenize(context)
    for i, sent in enumerate(sentences):
        if answer.replace(",", "") in sent.replace(",", ""):
            start = max(0, i - window)
            end = min(len(sentences), i + window + 1)
            return " ".join(sentences[start:end])
    return " ".join(sentences[:2])  # fallback