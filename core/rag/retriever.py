from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss
import pickle
from pathlib import Path
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

@dataclass
class RetrievedDocument:
    """Represents a retrieved document with its metadata and scores."""
    content: str
    metadata: Dict[str, Any]
    dense_score: float = 0.0
    sparse_score: float = 0.0
    combined_score: float = 0.0
    
    def __post_init__(self):
        if not self.combined_score:
            self.combined_score = (self.dense_score + self.sparse_score) / 2

@dataclass
class RetrieverConfig:
    """Configuration for the hybrid retriever."""
    dense_model_name: str = 'all-MiniLM-L6-v2'
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    cache_dir: str = '.cache/retriever'
    use_gpu: bool = False
    top_k: int = 5
    min_score: float = 0.3
    index_batch_size: int = 1000

class HybridRetriever:
    """A hybrid retriever that combines dense and sparse retrieval methods."""
    
    def __init__(self, config: RetrieverConfig = None):
        self.config = config or RetrieverConfig()
        self.dense_model = self._load_dense_model()
        self.document_store: Dict[int, Dict] = {}
        self.document_texts: List[str] = []
        self.document_embeddings = None
        self.bm25 = None
        self._setup_cache_dir()
        
    def _setup_cache_dir(self):
        """Create cache directory if it doesn't exist."""
        cache_path = Path(self.config.cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        
    def _load_dense_model(self) -> SentenceTransformer:
        """Load the dense retrieval model."""
        device = 'cuda' if self.config.use_gpu else 'cpu'
        return SentenceTransformer(self.config.dense_model_name, device=device)
    
    @lru_cache(maxsize=1000)
    def _get_dense_embeddings(self, text: str) -> np.ndarray:
        """Get dense embeddings for a text with caching."""
        return self.dense_model.encode([text])[0]
    
    def add_documents(self, documents: List[Dict[str, Any]], batch_size: Optional[int] = None):
        """Add documents to the retriever.
        
        Args:
            documents: List of document dictionaries with 'content' and 'metadata' keys
            batch_size: Optional batch size for processing
        """
        batch_size = batch_size or self.config.index_batch_size
        
        # Process documents in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            self._process_document_batch(batch)
            
        # Build BM25 index
        self.bm25 = BM25Okapi([doc.split() for doc in self.document_texts])
        
        # Save state
        self._save_state()
    
    def _process_document_batch(self, documents: List[Dict[str, Any]]):
        """Process a batch of documents."""
        texts = [doc['content'] for doc in documents]
        embeddings = self.dense_model.encode(texts, show_progress_bar=False)
        
        # Update document store and texts
        start_idx = len(self.document_store)
        for idx, (doc, embedding) in enumerate(zip(documents, embeddings), start_idx):
            self.document_store[idx] = {
                'content': doc['content'],
                'metadata': doc['metadata'],
                'embedding': embedding
            }
            self.document_texts.append(doc['content'])
            
        # Update FAISS index
        if self.document_embeddings is None:
            self.document_embeddings = embeddings
        else:
            self.document_embeddings = np.vstack([self.document_embeddings, embeddings])
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievedDocument]:
        """Retrieve documents using hybrid search.
        
        Args:
            query: The search query
            top_k: Optional number of results to return
            
        Returns:
            List of RetrievedDocument objects
        """
        top_k = top_k or self.config.top_k
        
        # Get dense scores
        query_embedding = self._get_dense_embeddings(query)
        dense_scores = self._get_dense_scores(query_embedding)
        
        # Get sparse scores
        sparse_scores = self._get_sparse_scores(query)
        
        # Combine scores
        combined_scores = self._combine_scores(dense_scores, sparse_scores)
        
        # Get top documents
        top_indices = np.argsort(combined_scores)[-top_k:]
        results = []
        
        for idx in top_indices:
            doc = self.document_store[idx]
            if combined_scores[idx] < self.config.min_score:
                continue
                
            results.append(RetrievedDocument(
                content=doc['content'],
                metadata=doc['metadata'],
                dense_score=dense_scores[idx],
                sparse_score=sparse_scores[idx],
                combined_score=combined_scores[idx]
            ))
            
        return list(reversed(results))
    
    def _get_dense_scores(self, query_embedding: np.ndarray) -> np.ndarray:
        """Calculate dense retrieval scores."""
        scores = np.dot(self.document_embeddings, query_embedding)
        return scores * self.config.dense_weight
    
    def _get_sparse_scores(self, query: str) -> np.ndarray:
        """Calculate sparse retrieval scores."""
        scores = np.array(self.bm25.get_scores(query.split()))
        return scores * self.config.sparse_weight
    
    def _combine_scores(self, dense_scores: np.ndarray, sparse_scores: np.ndarray) -> np.ndarray:
        """Combine dense and sparse scores."""
        # Normalize scores to [0, 1] range
        if len(dense_scores) > 0:
            dense_scores = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min() + 1e-6)
        if len(sparse_scores) > 0:
            sparse_scores = (sparse_scores - sparse_scores.min()) / (sparse_scores.max() - sparse_scores.min() + 1e-6)
        
        return dense_scores + sparse_scores
    
    def _save_state(self):
        """Save retriever state to disk."""
        state = {
            'document_store': self.document_store,
            'document_texts': self.document_texts,
            'document_embeddings': self.document_embeddings,
            'bm25': self.bm25
        }
        
        cache_path = Path(self.config.cache_dir)
        with open(cache_path / 'retriever_state.pkl', 'wb') as f:
            pickle.dump(state, f)
    
    def load_state(self):
        """Load retriever state from disk."""
        cache_path = Path(self.config.cache_dir)
        state_path = cache_path / 'retriever_state.pkl'
        
        if not state_path.exists():
            logger.warning(f'No saved state found at {state_path}')
            return
            
        with open(state_path, 'rb') as f:
            state = pickle.load(f)
            
        self.document_store = state['document_store']
        self.document_texts = state['document_texts']
        self.document_embeddings = state['document_embeddings']
        self.bm25 = state['bm25']
        
    def clear_cache(self):
        """Clear the retriever cache."""
        cache_path = Path(self.config.cache_dir)
        if cache_path.exists():
            for file in cache_path.glob('*'):
                file.unlink()
            cache_path.rmdir()
        self._setup_cache_dir()