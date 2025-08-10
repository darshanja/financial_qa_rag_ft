from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Union

class HybridRetriever:
    def __init__(self, chunks: List[Union[str, Dict]], embedder):
        self.chunks = chunks
        self.embedder = embedder
        
        # Handle both string chunks and dict chunks
        if chunks and isinstance(chunks[0], dict):
            self.texts = [c['text'] for c in chunks]
        else:
            self.texts = chunks
            
        self.embeddings = embedder.encode(self.texts)
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(self.texts)

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """
        Retrieve most relevant chunks using hybrid search.
        
        Args:
            query: Search query
            top_k: Number of chunks to retrieve
            
        Returns:
            List of most relevant text chunks
        """
        # Get dense embeddings
        query_embedding = self.embedder.encode([query])
        dense_scores = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get sparse scores
        sparse_query = self.tfidf.transform([query])
        sparse_scores = cosine_similarity(sparse_query, self.tfidf_matrix)[0]
        
        # Combine scores (weighted average)
        combined_scores = 0.7 * dense_scores + 0.3 * sparse_scores
        
        # Get top chunks
        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        
        if isinstance(self.chunks[0], dict):
            return [self.chunks[i]['text'] for i in top_indices]
        else:
            return [self.chunks[i] for i in top_indices]
