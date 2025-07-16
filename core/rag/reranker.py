from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
from sentence_transformers import CrossEncoder
from tqdm import tqdm

@dataclass
class RerankedResult:
    """Container for reranked search results."""
    text: str
    score: float
    metadata: dict
    original_rank: int

class SemanticReranker:
    """Reranks search results using semantic similarity with cross-encoder models."""

    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
                 batch_size: int = 32,
                 use_gpu: bool = True):
        """Initialize the reranker.

        Args:
            model_name: Name of the cross-encoder model to use
            batch_size: Batch size for inference
            use_gpu: Whether to use GPU acceleration if available
        """
        self.model = CrossEncoder(model_name, device='cuda' if use_gpu else 'cpu')
        self.batch_size = batch_size

    def rerank(self,
               query: str,
               documents: List[str],
               metadata: Optional[List[dict]] = None,
               top_k: Optional[int] = None,
               score_threshold: Optional[float] = None) -> List[RerankedResult]:
        """Rerank documents based on semantic similarity to query.

        Args:
            query: Search query
            documents: List of document texts to rerank
            metadata: Optional list of metadata dicts for each document
            top_k: Number of top results to return (default: all)
            score_threshold: Minimum score threshold for filtering results

        Returns:
            List of RerankedResult objects sorted by relevance score
        """
        if not documents:
            return []

        # Prepare cross-encoder inputs
        pairs = [[query, doc] for doc in documents]
        metadata = metadata or [{}] * len(documents)

        # Get relevance scores in batches
        all_scores = []
        for i in tqdm(range(0, len(pairs), self.batch_size)):
            batch = pairs[i:i + self.batch_size]
            scores = self.model.predict(batch)
            all_scores.extend(scores)

        # Create results with scores and metadata
        results = [
            RerankedResult(
                text=doc,
                score=float(score),
                metadata=meta,
                original_rank=idx
            )
            for idx, (doc, score, meta) in enumerate(zip(documents, all_scores, metadata))
        ]

        # Sort by score in descending order
        results.sort(key=lambda x: x.score, reverse=True)

        # Apply score threshold if specified
        if score_threshold is not None:
            results = [r for r in results if r.score >= score_threshold]

        # Limit to top_k if specified
        if top_k is not None:
            results = results[:top_k]

        return results

    def rerank_with_diversity(self,
                            query: str,
                            documents: List[str],
                            metadata: Optional[List[dict]] = None,
                            top_k: int = 10,
                            diversity_threshold: float = 0.8,
                            max_similarity: float = 0.95) -> List[RerankedResult]:
        """Rerank documents with diversity-aware selection.

        Args:
            query: Search query
            documents: List of document texts to rerank
            metadata: Optional list of metadata dicts for each document
            top_k: Number of diverse results to return
            diversity_threshold: Threshold for considering documents diverse
            max_similarity: Maximum allowed similarity between selected documents

        Returns:
            List of diverse RerankedResult objects
        """
        # First get initial ranking
        initial_results = self.rerank(query, documents, metadata)

        if not initial_results or top_k <= 1:
            return initial_results[:top_k] if initial_results else []

        # Initialize diverse results with the top-scoring document
        diverse_results = [initial_results[0]]
        candidates = initial_results[1:]

        while len(diverse_results) < top_k and candidates:
            # Get embeddings for diversity comparison
            candidate_texts = [c.text for c in candidates]
            selected_texts = [d.text for d in diverse_results]

            # Calculate similarities between candidates and selected documents
            similarities = self._calculate_similarities(candidate_texts, selected_texts)

            # Find most diverse candidate
            max_similarities = np.max(similarities, axis=1)
            diverse_idx = np.argmin(max_similarities)

            # Check if candidate is diverse enough
            if max_similarities[diverse_idx] < max_similarity:
                diverse_results.append(candidates[diverse_idx])
                candidates.pop(diverse_idx)
            else:
                # No sufficiently diverse candidates found
                break

        return diverse_results

    def _calculate_similarities(self, texts1: List[str], texts2: List[str]) -> np.ndarray:
        """Calculate pairwise similarities between two lists of texts."""
        pairs = [(t1, t2) for t1 in texts1 for t2 in texts2]
        similarities = self.model.predict(pairs)
        return similarities.reshape(len(texts1), len(texts2))

    @staticmethod
    def combine_scores(relevance_score: float,
                      diversity_score: float,
                      alpha: float = 0.7) -> float:
        """Combine relevance and diversity scores.

        Args:
            relevance_score: Base relevance score
            diversity_score: Diversity score
            alpha: Weight for relevance score (1-alpha for diversity)

        Returns:
            Combined score balancing relevance and diversity
        """
        return alpha * relevance_score + (1 - alpha) * diversity_score