

import json
import numpy as np
from typing import List, Dict
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


def _simple_tokenize(text: str) -> List[str]:
    """Lightweight tokenizer for BM25 — lowercase + split on non-alpha.
    Good enough for English prose; swap for a proper tokenizer if needed."""
    import re
    return re.findall(r"[a-z0-9']+", text.lower())


class Retriever:
    def __init__(
        self,
        chunks: List[Dict],
        embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        """
        chunks: list of {"chunk_id", "book", "page_start", "page_end", "text", ...}
        """
        self.chunks = chunks
        self.texts = [c["text"] for c in chunks]

        print(f"Building BM25 index over {len(chunks)} chunks...")
        tokenized = [_simple_tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)

        print(f"Loading embedding model: {embed_model_name}")
        self.embed_model = SentenceTransformer(embed_model_name)
        print("Encoding chunk embeddings...")
        self.chunk_embeddings = self.embed_model.encode(
            self.texts, show_progress_bar=True, normalize_embeddings=True
        )

        print(f"Loading reranker: {reranker_model_name}")
        self.reranker = CrossEncoder(reranker_model_name)

        print("Retriever ready.\n")

    def _bm25_search(self, query: str, top_n: int) -> List[int]:
        tokenized_query = _simple_tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_idx = np.argsort(scores)[::-1][:top_n]
        return [i for i in top_idx if scores[i] > 0]

    def _embedding_search(self, query: str, top_n: int) -> List[int]:
        q_emb = self.embed_model.encode([query], normalize_embeddings=True)[0]
        sims = self.chunk_embeddings @ q_emb  # cosine sim, since normalized
        top_idx = np.argsort(sims)[::-1][:top_n]
        return list(top_idx)

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_pool: int = 20,
    ) -> List[Dict]:
        """
        Hybrid retrieval + rerank.

        1. Get top `candidate_pool` from BM25 and from embeddings separately.
        2. Merge + dedupe candidate indices.
        3. Rerank the merged set with the cross-encoder.
        4. Return top_k chunks with scores attached.
        """
        bm25_idx = self._bm25_search(query, candidate_pool)
        emb_idx = self._embedding_search(query, candidate_pool)

        candidate_idx = list(dict.fromkeys(bm25_idx + emb_idx))  # dedupe, preserve order
        if not candidate_idx:
            return []

        pairs = [(query, self.texts[i]) for i in candidate_idx]
        rerank_scores = self.reranker.predict(pairs)

        ranked = sorted(
            zip(candidate_idx, rerank_scores), key=lambda x: x[1], reverse=True
        )[:top_k]

        results = []
        for idx, score in ranked:
            chunk = dict(self.chunks[idx])
            chunk["rerank_score"] = float(score)
            chunk["in_bm25_candidates"] = idx in bm25_idx
            chunk["in_embedding_candidates"] = idx in emb_idx
            results.append(chunk)

        return results
