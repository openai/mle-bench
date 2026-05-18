"""
Hybrid Retriever for Task Memory

Combines BM25 keyword search with vector semantic search using RRF fusion.
"""

import logging
from typing import List, Tuple, Optional, TYPE_CHECKING, Any
import numpy as np
import faiss
from rank_bm25 import BM25Okapi

if TYPE_CHECKING:
    from .embedding_models import EmbeddingModel

logger = logging.getLogger("aide")


class HybridRetriever:
    """
    Hybrid retriever combining BM25 and vector search

    Uses Reciprocal Rank Fusion (RRF) to combine results
    """

    def __init__(self, embedding_model: "EmbeddingModel"):
        """
        Initialize hybrid retriever

        Args:
            embedding_model: EmbeddingModel instance for vector search
        """
        self.embedding_model = embedding_model
        self.dimension = embedding_model.dimension

        # BM25 components
        self.bm25: Optional[BM25Okapi] = None
        self.bm25_corpus: List[List[str]] = []

        # Vector components
        self.vector_index: Optional[faiss.IndexFlatL2] = None
        self.vectors: Optional[np.ndarray] = None

        # Data
        self.records: List[Any] = []
        self.texts: List[str] = []

    def build_index(self, records: List[Any], texts: List[str]):
        """
        Build both BM25 and vector indexes

        Args:
            records: List of record objects
            texts: List of text strings (corresponding to records)
        """
        self.records = records
        self.texts = texts

        # Build BM25 index
        tokenized_corpus = [text.lower().split() for text in texts]
        self.bm25_corpus = tokenized_corpus
        self.bm25 = BM25Okapi(tokenized_corpus)

        # Build vector index
        logger.info(f"Computing embeddings for {len(texts)} texts...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        self.vectors = embeddings
        self.vector_index = faiss.IndexFlatL2(self.dimension)
        self.vector_index.add(embeddings)
        logger.info(f"Built vector index with {len(texts)} vectors")

    def save_index(self, embeddings_path: str, index_path: str):
        """
        Save embeddings and FAISS index to disk

        Args:
            embeddings_path: Path to save embeddings (.npy)
            index_path: Path to save FAISS index
        """
        if self.vectors is None or self.vector_index is None:
            raise ValueError("No index to save. Build index first.")

        # Save embeddings as numpy array
        np.save(embeddings_path, self.vectors)

        # Save FAISS index
        faiss.write_index(self.vector_index, index_path)

    def load_index(self, records: List[Any], texts: List[str], embeddings_path: str, index_path: str):
        """
        Load embeddings and FAISS index from disk

        Args:
            records: List of record objects
            texts: List of text strings (corresponding to records)
            embeddings_path: Path to load embeddings (.npy)
            index_path: Path to load FAISS index
        """
        self.records = records
        self.texts = texts

        # Load embeddings
        self.vectors = np.load(embeddings_path)

        # Load FAISS index
        self.vector_index = faiss.read_index(index_path)

        # Validate dimensions
        if self.vectors.shape[0] != len(records):
            raise ValueError(f"Embeddings count ({self.vectors.shape[0]}) doesn't match records count ({len(records)})")

        if self.vectors.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension ({self.vectors.shape[1]}) doesn't match expected ({self.dimension})")

        # Rebuild BM25 index (lightweight, always rebuild)
        tokenized_corpus = [text.lower().split() for text in texts]
        self.bm25_corpus = tokenized_corpus
        self.bm25 = BM25Okapi(tokenized_corpus)

    def add_to_index(self, new_records: List[Any], new_texts: List[str]):
        """
        Add new records to existing index (incremental update)

        Args:
            new_records: New record objects to add
            new_texts: New text strings (corresponding to new_records)
        """
        if not new_records or not new_texts:
            return

        # Compute embeddings for new texts only
        logger.info(f"Computing embeddings for {len(new_texts)} new texts...")
        new_embeddings = self.embedding_model.encode(new_texts, show_progress_bar=len(new_texts) > 5)

        # Note: Don't extend self.records here!
        # The records list is shared with the memory layer, which has already added the new records.
        # We only need to extend texts (which is local to retriever)
        self.texts.extend(new_texts)

        # Update vector index
        if self.vectors is not None:
            self.vectors = np.vstack([self.vectors, new_embeddings])
        else:
            self.vectors = new_embeddings

        if self.vector_index is None:
            self.vector_index = faiss.IndexFlatL2(self.dimension)

        self.vector_index.add(new_embeddings)

        # Rebuild BM25 index (always rebuild, it's fast)
        tokenized_corpus = [text.lower().split() for text in self.texts]
        self.bm25_corpus = tokenized_corpus
        self.bm25 = BM25Okapi(tokenized_corpus)

        logger.info(f"Added {len(new_texts)} new vectors to index (total: {len(self.texts)})")

    def search(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.5,
        bm25_top_k: int = 50,
        vector_top_k: int = 50,
        rrf_k: int = 60,
        return_scores: bool = False
    ) -> List[Tuple[Any, float]]:
        """
        Hybrid search using BM25 and vector search with RRF fusion

        Args:
            query: Query string
            top_k: Number of final results to return
            alpha: Weight for BM25 vs vector. Range [0, 1]
                   alpha=1.0: pure BM25, alpha=0.0: pure vector, alpha=0.5: balanced
            bm25_top_k: Number of candidates from BM25 search
            vector_top_k: Number of candidates from vector search
            rrf_k: RRF constant (default 60)
            return_scores: Whether to return similarity scores

        Returns:
            List of (record, score) tuples, sorted by RRF score (descending)
        """
        if not self.records:
            return []

        # BM25 search
        bm25_results = []
        if self.bm25 and alpha > 0:
            tokenized_query = query.lower().split()
            bm25_scores = self.bm25.get_scores(tokenized_query)

            # Get top-k BM25 results
            bm25_results = sorted(
                zip(self.records, bm25_scores),
                key=lambda x: x[1],
                reverse=True
            )[:bm25_top_k]

        # Vector search
        vector_results = []
        if self.vector_index and (1 - alpha) > 0:
            query_embedding = self.embedding_model.encode([query])
            distances, indices = self.vector_index.search(query_embedding, min(vector_top_k, len(self.records)))

            # Convert L2 distance to similarity score
            # Use negative distance as score (smaller distance = higher score)
            vector_results = [
                (self.records[idx], -float(dist))
                for idx, dist in zip(indices[0], distances[0])
            ]

        # Fusion strategy based on alpha
        if alpha == 1.0:
            # Pure BM25
            final_results = bm25_results[:top_k]
        elif alpha == 0.0:
            # Pure vector
            final_results = vector_results[:top_k]
        elif not bm25_results:
            # BM25 failed, use vector only
            final_results = vector_results[:top_k]
        elif not vector_results:
            # Vector failed, use BM25 only
            final_results = bm25_results[:top_k]
        else:
            # Hybrid: Use RRF fusion (Reciprocal Rank Fusion)
            # Reference: EverMemOS retrieval_utils.py:reciprocal_rank_fusion
            final_results = self._reciprocal_rank_fusion(
                bm25_results,
                vector_results,
                k=rrf_k,
                alpha=alpha
            )[:top_k]

        return final_results

    def _reciprocal_rank_fusion(
        self,
        results1: List[Tuple],
        results2: List[Tuple],
        k: int = 60,
        alpha: float = 0.5
    ) -> List[Tuple]:
        """
        RRF fusion of two result lists with alpha weighting

        Formula: RRF_score = alpha * (1 / (k + rank1)) + (1 - alpha) * (1 / (k + rank2))

        Args:
            results1: First result list (e.g., BM25)
            results2: Second result list (e.g., Vector)
            k: RRF constant (default 60)
            alpha: Weight for results1 (1-alpha for results2)

        Returns:
            Fused results sorted by RRF score (descending)
        """
        doc_rrf_scores = {}
        doc_map = {}

        # Process first result set (weighted by alpha)
        for rank, (doc, score) in enumerate(results1, start=1):
            doc_id = id(doc)
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
            doc_rrf_scores[doc_id] = doc_rrf_scores.get(doc_id, 0.0) + alpha * (1.0 / (k + rank))

        # Process second result set (weighted by 1-alpha)
        for rank, (doc, score) in enumerate(results2, start=1):
            doc_id = id(doc)
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
            doc_rrf_scores[doc_id] = doc_rrf_scores.get(doc_id, 0.0) + (1.0 - alpha) * (1.0 / (k + rank))

        # Sort by RRF score
        fused_results = [
            (doc_map[doc_id], rrf_score)
            for doc_id, rrf_score in doc_rrf_scores.items()
        ]
        fused_results.sort(key=lambda x: x[1], reverse=True)

        return fused_results
