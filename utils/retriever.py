from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class HybridRetriever:
    def __init__(self, chunks, embedder):
        self.chunks = chunks
        self.embedder = embedder
        self.texts = [c['text'] for c in chunks]
        self.embeddings = embedder.encode(self.texts, convert_to_tensor=True)
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(self.texts)

    def retrieve(self, query, top_k=5):
        dense_query = self.embedder.encode([query], convert_to_tensor=True)
        sparse_query = self.tfidf.transform([query])

        dense_scores = cosine_similarity(dense_query, self.embeddings)[0]
        sparse_scores = cosine_similarity(sparse_query, self.tfidf_matrix)[0]

        combined_scores = (dense_scores + sparse_scores) / 2
        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        return [self.chunks[i] for i in top_indices]
