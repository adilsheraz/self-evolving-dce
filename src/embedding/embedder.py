"""
Embedding providers for text vectorization
Supports HuggingFace Inference API with various models
"""
import numpy as np
from typing import List
from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers"""
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        pass
    
    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query text
        
        Args:
            query: Query string
            
        Returns:
            numpy array of shape (embedding_dim,)
        """
        pass


class HuggingFaceEmbedder(EmbeddingProvider):
    """
    HuggingFace Inference API embedding provider
    Supports BGE models and other sentence transformers
    """
    
    def __init__(
        self,
        api_key: str = None,
        model: str = "BAAI/bge-small-en-v1.5",
        query_prefix: str = None
    ):
        """
        Initialize HuggingFace embedder
        
        Args:
            api_key: HuggingFace API token (or uses HF_TOKEN env var)
            model: Model identifier on HuggingFace Hub
            query_prefix: Optional prefix for query texts (BGE models use special prefixes)
        """
        from huggingface_hub import InferenceClient
        from src.config import Config
        
        # Use provided key or fall back to config
        self.api_key = api_key or Config.HF_TOKEN
        if not self.api_key:
            raise ValueError("HF_TOKEN not provided and not found in environment")
        
        self.model = model
        self.client = InferenceClient(
            provider="hf-inference",
            api_key=self.api_key
        )
        
        # BGE models use a special query prefix
        if query_prefix is None and "bge" in model.lower():
            self.query_prefix = "Represent this sentence for searching relevant passages: "
        else:
            self.query_prefix = query_prefix or ""
    
    def _prepare_text(self, text: str, is_query: bool = False) -> str:
        """Add prefix to text if needed"""
        if is_query and self.query_prefix:
            return self.query_prefix + text
        return text
    
    def embed_texts(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """
        Embed multiple texts
        
        Args:
            texts: List of texts to embed
            is_query: Whether these are query texts (affects prefix)
            
        Returns:
            numpy array of embeddings (len(texts), embedding_dim)
        """
        embeddings = []
        for text in texts:
            prepared = self._prepare_text(text, is_query=is_query)
            emb = self.client.feature_extraction(
                prepared,
                model=self.model
            )
            embeddings.append(emb)
        
        return np.array(embeddings, dtype=np.float32)
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query
        
        Args:
            query: Query text
            
        Returns:
            numpy array of shape (embedding_dim,)
        """
        return self.embed_texts([query], is_query=True)[0]
    
    def embed_in_batches(
        self,
        texts: List[str],
        batch_size: int = 16,
        is_query: bool = False,
        verbose: bool = True
    ) -> np.ndarray:
        """
        Embed texts in batches for efficiency
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts per batch
            is_query: Whether these are query texts
            verbose: Print progress
            
        Returns:
            numpy array of all embeddings
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            if verbose:
                print(f"🔢 Embedding batch {i} → {i + len(batch)} / {len(texts)}")
            
            emb = self.embed_texts(batch, is_query=is_query)
            all_embeddings.append(emb)
        
        return np.vstack(all_embeddings)
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension (requires a test query)"""
        test_emb = self.embed_query("test")
        return len(test_emb)


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    L2 normalize embeddings for cosine similarity
    
    Args:
        embeddings: array of shape (n, dim) or (dim,)
        
    Returns:
        Normalized embeddings
    """
    if embeddings.ndim == 1:
        norm = np.linalg.norm(embeddings)
        return embeddings / (norm + 1e-12)
    
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / (norms + 1e-12)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Cosine similarity score
    """
    a_norm = a / (np.linalg.norm(a) + 1e-12)
    b_norm = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a_norm, b_norm))
