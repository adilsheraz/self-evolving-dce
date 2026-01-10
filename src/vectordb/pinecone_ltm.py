"""
Pinecone Long-Term Memory (LTM) Integration
Persistent vector storage for DCE system
"""
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path


class PineconeLTM:
    """
    Pinecone-based Long-Term Memory for DCE
    
    Namespaces:
    - kb_chunks: Knowledge base chunks
    - dynamic_contexts: Composed contexts
    - context_evaluations: Evaluation results
    - agent_experiments: Agent decision logs
    """
    
    # Namespace constants
    NS_CHUNKS = "kb_chunks"
    NS_CONTEXTS = "dynamic_contexts"
    NS_EVALS = "context_evaluations"
    NS_EXPERIMENTS = "agent_experiments"
    
    def __init__(
        self,
        api_key: str = None,
        index_name: str = "self-evolving-dce-ltm",
        embedding_dim: int = 384,
        metric: str = "cosine",
        cloud: str = "aws",
        region: str = "us-east-1"
    ):
        """
        Initialize Pinecone LTM
        
        Args:
            api_key: Pinecone API key (or uses PINECONE_API_KEY from config)
            index_name: Name of Pinecone index
            embedding_dim: Dimension of embeddings (384 for BGE-small)
            metric: Distance metric (cosine, euclidean, dotproduct)
            cloud: Cloud provider (aws, gcp, azure)
            region: Cloud region
        """
        from pinecone import Pinecone, ServerlessSpec
        from src.config import Config
        
        # Use provided key or fall back to config
        self.api_key = api_key or Config.PINECONE_API_KEY
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY not provided and not found in environment")
        
        self.index_name = index_name
        self.embedding_dim = embedding_dim
        self.metric = metric
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=self.api_key)
        
        # Create or connect to index
        self.spec = ServerlessSpec(cloud=cloud, region=region)
        self._ensure_index()
        
        # Connect to index
        self.index = self.pc.Index(self.index_name)
    
    def _ensure_index(self) -> None:
        """Create index if it doesn't exist"""
        existing = [idx["name"] for idx in self.pc.list_indexes()]
        
        if self.index_name not in existing:
            self.pc.create_index(
                name=self.index_name,
                dimension=self.embedding_dim,
                metric=self.metric,
                spec=self.spec
            )
            print(f"✅ Created Pinecone index: {self.index_name}")
        else:
            print(f"✅ Connected to existing Pinecone index: {self.index_name}")
    
    def upsert_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: np.ndarray,
        batch_size: int = 200,
        namespace: str = None
    ) -> int:
        """
        Upsert knowledge chunks to Pinecone
        
        Args:
            chunks: List of chunk dictionaries with metadata
            embeddings: Array of embeddings (len == len(chunks))
            batch_size: Batch size for upsert operations
            namespace: Namespace to use (default: NS_CHUNKS)
            
        Returns:
            Number of chunks upserted
        """
        namespace = namespace or self.NS_CHUNKS
        
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) length mismatch"
            )
        
        vectors = []
        for i, chunk in enumerate(chunks):
            vec = embeddings[i]
            vec_list = vec.tolist() if hasattr(vec, "tolist") else list(vec)
            
            if len(vec_list) != self.embedding_dim:
                raise ValueError(
                    f"Embedding dimension mismatch at index {i}: "
                    f"got {len(vec_list)}, expected {self.embedding_dim}"
                )
            
            vectors.append({
                "id": str(chunk.get("chunk_id", f"chunk_{i}")),
                "values": vec_list,
                "metadata": {
                    "text": str(chunk.get("text", chunk.get("content", ""))),
                    "doc_id": str(chunk.get("doc_id", "")),
                    "title": str(chunk.get("title", "")),
                    "heading": str(chunk.get("heading", "")),
                    "tokens": int(chunk.get("tokens", 0)),
                    "source_path": str(chunk.get("source_path", "")),
                    "type": "kb_chunk"
                }
            })
        
        # Batch upsert
        total_upserted = 0
        for start in range(0, len(vectors), batch_size):
            batch = vectors[start:start + batch_size]
            self.index.upsert(vectors=batch, namespace=namespace)
            total_upserted += len(batch)
            print(f"📤 Upserted batch {start} → {start + len(batch)} / {len(vectors)}")
        
        print(f"✅ Upserted {total_upserted} chunks to namespace={namespace}")
        return total_upserted
    
    def query(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        namespace: str = None,
        filter_dict: Dict[str, Any] = None,
        include_metadata: bool = True,
        include_values: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Query Pinecone index with a vector
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            namespace: Namespace to query (default: NS_CHUNKS)
            filter_dict: Metadata filters
            include_metadata: Include metadata in results
            include_values: Include embedding values in results
            
        Returns:
            List of matches with scores and metadata
        """
        namespace = namespace or self.NS_CHUNKS
        
        vec_list = query_vector.tolist() if hasattr(query_vector, "tolist") else list(query_vector)
        
        results = self.index.query(
            vector=vec_list,
            top_k=top_k,
            namespace=namespace,
            filter=filter_dict,
            include_metadata=include_metadata,
            include_values=include_values
        )
        
        return results.get("matches", [])
    
    def delete_namespace(self, namespace: str) -> None:
        """Delete all vectors in a namespace"""
        self.index.delete(delete_all=True, namespace=namespace)
        print(f"✅ Deleted all vectors in namespace={namespace}")
    
    def get_stats(self, namespace: str = None) -> Dict[str, Any]:
        """Get index statistics"""
        stats = self.index.describe_index_stats()
        
        if namespace:
            ns_stats = stats.get("namespaces", {}).get(namespace, {})
            return {
                "namespace": namespace,
                "vector_count": ns_stats.get("vector_count", 0)
            }
        
        return {
            "total_vector_count": stats.get("total_vector_count", 0),
            "dimension": stats.get("dimension", 0),
            "namespaces": {
                ns: info.get("vector_count", 0)
                for ns, info in stats.get("namespaces", {}).items()
            }
        }
    
    def upsert_context(
        self,
        context_id: str,
        context_embedding: np.ndarray,
        metadata: Dict[str, Any],
        namespace: str = None
    ) -> None:
        """
        Upsert a composed dynamic context
        
        Args:
            context_id: Unique context identifier
            context_embedding: Context embedding vector
            metadata: Context metadata (query, chunks used, scores, etc.)
            namespace: Namespace (default: NS_CONTEXTS)
        """
        namespace = namespace or self.NS_CONTEXTS
        
        vec_list = (context_embedding.tolist() 
                   if hasattr(context_embedding, "tolist") 
                   else list(context_embedding))
        
        self.index.upsert(
            vectors=[{
                "id": context_id,
                "values": vec_list,
                "metadata": {**metadata, "type": "dynamic_context"}
            }],
            namespace=namespace
        )
    
    def upsert_evaluation(
        self,
        eval_id: str,
        eval_embedding: np.ndarray,
        metadata: Dict[str, Any],
        namespace: str = None
    ) -> None:
        """
        Upsert a context evaluation result
        
        Args:
            eval_id: Unique evaluation identifier
            eval_embedding: Evaluation embedding (e.g., query embedding)
            metadata: Evaluation scores and metrics
            namespace: Namespace (default: NS_EVALS)
        """
        namespace = namespace or self.NS_EVALS
        
        vec_list = (eval_embedding.tolist() 
                   if hasattr(eval_embedding, "tolist") 
                   else list(eval_embedding))
        
        self.index.upsert(
            vectors=[{
                "id": eval_id,
                "values": vec_list,
                "metadata": {**metadata, "type": "evaluation"}
            }],
            namespace=namespace
        )
    
    def upsert_experiment(
        self,
        experiment_id: str,
        experiment_embedding: np.ndarray,
        metadata: Dict[str, Any],
        namespace: str = None
    ) -> None:
        """
        Upsert an agent experiment log
        
        Args:
            experiment_id: Unique experiment identifier
            experiment_embedding: Experiment embedding
            metadata: Experiment data (policy, results, decisions)
            namespace: Namespace (default: NS_EXPERIMENTS)
        """
        namespace = namespace or self.NS_EXPERIMENTS
        
        vec_list = (experiment_embedding.tolist() 
                   if hasattr(experiment_embedding, "tolist") 
                   else list(experiment_embedding))
        
        self.index.upsert(
            vectors=[{
                "id": experiment_id,
                "values": vec_list,
                "metadata": {**metadata, "type": "experiment"}
            }],
            namespace=namespace
        )
