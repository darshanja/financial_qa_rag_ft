import time
from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

def calculate_confidence(answer: str) -> float:
    """
    Calculate confidence score for generated answer.
    """
    confidence = 1.0
    
    # Reduce confidence for uncertain language
    uncertainty_markers = [
        "might", "may", "could", "possibly", "perhaps",
        "I think", "probably", "likely", "seems", "appears"
    ]
    for marker in uncertainty_markers:
        if marker in answer.lower():
            confidence *= 0.9
            
    # Reduce confidence for very short or very long answers
    words = answer.split()
    if len(words) < 5:
        confidence *= 0.8
    elif len(words) > 100:
        confidence *= 0.9
        
    # Reduce confidence if answer contains non-financial terms
    financial_terms = [
        "revenue", "profit", "loss", "income", "expense",
        "asset", "liability", "equity", "cash", "stock",
        "share", "dividend", "market", "financial", "fiscal"
    ]
    if not any(term in answer.lower() for term in financial_terms):
        confidence *= 0.7
        
    return max(0.1, min(confidence, 1.0))

def calculate_semantic_similarity(text1: str, text2: str) -> float:
    """
    Calculate semantic similarity between two texts using sentence embeddings.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode([text1, text2])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return float(similarity)

def evaluate_response(query: str, answer: str, chunks: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Evaluate the quality of the generated response.
    """
    confidence = calculate_confidence(answer)
    
    metrics = {
        "confidence": confidence,
        "answer_length": len(answer.split()),
        "query_length": len(query.split()),
    }
    
    if chunks:
        metrics["num_chunks"] = len(chunks)
        # Calculate chunk relevance score
        if len(chunks) > 0:
            # Split query into terms, excluding common words
            query_terms = [term.lower() for term in query.split() 
                         if term.lower() not in {'what', 'was', 'is', 'are', 'in', 'the', 'a', 'an', 'and', 'or'}]
            
            # Calculate relevance for each chunk
            chunk_scores = []
            for chunk in chunks:
                chunk_lower = chunk.lower()
                matches = sum(1 for term in query_terms if term in chunk_lower)
                chunk_scores.append(matches / len(query_terms) if query_terms else 0)
                
            # Take average of chunk scores
            metrics["chunk_relevance"] = sum(chunk_scores) / len(chunks)
        else:
            metrics["chunk_relevance"] = 0.0
        
    return metrics

def evaluate_models(questions: List[str], answers: List[str], rag_fn, ft_fn) -> List[Dict]:
    """
    Evaluate and compare RAG and fine-tuned models.
    """
    results = []
    for q, a in zip(questions, answers):
        start = time.time()
        rag_answer = rag_fn(q)
        rag_time = time.time() - start

        start = time.time()
        ft_answer = ft_fn(q)
        ft_time = time.time() - start

        results.append({
            "question": q,
            "ground_truth": a,
            "rag_answer": rag_answer,
            "rag_time": round(rag_time, 2),
            "ft_answer": ft_answer,
            "ft_time": round(ft_time, 2)
        })
    return results
