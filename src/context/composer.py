"""
Dynamic Context Composer

Greedy, budget-aware selection and ordering of chunks for context packages.

Key Features:
- Token budget management
- Redundancy filtering
- Document diversity enforcement
- Weighted scoring (retrieval, novelty, diversity)
- Deterministic given same inputs
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional


class ContextComposer:
    """
    Compose context packages from candidate chunks.
    
    Uses greedy selection with weighted scoring:
    - Retrieval relevance (similarity to query)
    - Novelty (dissimilarity to already-selected chunks)
    - Document diversity (spreading across source documents)
    """
    
    # Default policy parameters
    DEFAULT_POLICY = {
        "policy_version": "v0.1",
        "budget_tokens": 1200,
        "min_chunk_tokens": 80,
        "max_chunks": 10,
        "max_chunks_per_doc": 2,
        "w_retrieval": 0.55,
        "w_novelty": 0.25,
        "w_doc_diversity": 0.20,
        "redundancy_threshold": 0.88,
        "redundancy_penalty": 0.35,
        "order_mode": "retrieval_then_structure",
    }
    
    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        chunk_embeddings: np.ndarray,
        policy: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize context composer.
        
        Args:
            chunks: List of chunk dictionaries (must include: chunk_id, doc_id, title, tokens, text)
            chunk_embeddings: Normalized embeddings aligned with chunks
            policy: Policy parameters (uses defaults if not provided)
        """
        self.chunks = chunks
        self.chunk_embeddings = self._ensure_normalized(chunk_embeddings)
        self.policy = {**self.DEFAULT_POLICY, **(policy or {})}
        
        # Build chunk_id -> index mapping
        self.chunk_id_to_index = {c["chunk_id"]: i for i, c in enumerate(chunks)}
    
    @staticmethod
    def _ensure_normalized(embeddings: np.ndarray) -> np.ndarray:
        """Ensure embeddings are L2-normalized."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        return embeddings / norms
    
    @staticmethod
    def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        return float(np.dot(a, b))
    
    @staticmethod
    def _cos_sim_many(mat: np.ndarray, vec: np.ndarray) -> np.ndarray:
        """Cosine similarity between matrix rows and vector."""
        return mat @ vec
    
    def _retrieval_scores(self, candidate_idxs: List[int], query_vec: np.ndarray) -> np.ndarray:
        """Compute retrieval similarity scores for candidates."""
        cand_emb = self.chunk_embeddings[candidate_idxs]
        return self._cos_sim_many(cand_emb, query_vec)
    
    def _novelty_score(self, chunk_vec: np.ndarray, selected_vecs: np.ndarray) -> float:
        """
        Compute novelty score (diversity relative to selected chunks).
        
        Novelty = 1 - max_similarity_to_selected
        """
        if selected_vecs.size == 0:
            return 1.0
        sims = selected_vecs @ chunk_vec
        return float(1.0 - np.max(sims))
    
    def _doc_diversity_boost(
        self,
        doc_id: str,
        doc_counts: Dict[str, int],
        max_per_doc: int
    ) -> float:
        """
        Compute document diversity boost.
        
        Prefers documents not yet heavily used.
        """
        used = doc_counts.get(doc_id, 0)
        if used == 0:
            return 1.0
        return max(0.0, 1.0 - (used / max_per_doc))
    
    def _is_redundant(
        self,
        chunk_vec: np.ndarray,
        selected_vecs: np.ndarray,
        threshold: float
    ) -> bool:
        """Check if chunk is too similar to already-selected chunks."""
        if selected_vecs.size == 0:
            return False
        sims = selected_vecs @ chunk_vec
        return bool(np.max(sims) >= threshold)
    
    def candidates_to_indices(self, retriever_results: List[Dict[str, Any]]) -> List[int]:
        """
        Convert retriever results to chunk indices.
        
        Args:
            retriever_results: List of dicts with 'chunk_id' field
            
        Returns:
            List of indices into self.chunks
        """
        idxs = []
        for r in retriever_results:
            cid = r.get("chunk_id")
            if cid in self.chunk_id_to_index:
                idxs.append(self.chunk_id_to_index[cid])
        return idxs
    
    def compose(
        self,
        query: str,
        candidate_idxs: List[int],
        query_vec: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compose context package from candidate chunks.
        
        Args:
            query: Query text
            candidate_idxs: Indices of candidate chunks (from retriever)
            query_vec: Normalized query embedding
            
        Returns:
            Context package dictionary with selected chunks, metrics, and metadata
        """
        # Extract policy parameters
        budget = int(self.policy["budget_tokens"])
        max_chunks = int(self.policy["max_chunks"])
        max_per_doc = int(self.policy["max_chunks_per_doc"])
        red_thr = float(self.policy["redundancy_threshold"])
        
        w_ret = float(self.policy["w_retrieval"])
        w_nov = float(self.policy["w_novelty"])
        w_div = float(self.policy["w_doc_diversity"])
        
        # Initialize state
        selected = []
        excluded = []
        used_tokens = 0
        doc_counts: Dict[str, int] = {}
        selected_vecs = np.empty((0, self.chunk_embeddings.shape[1]), dtype=np.float32)
        
        # Precompute retrieval scores
        ret_sims = self._retrieval_scores(candidate_idxs, query_vec)
        
        # Candidates to process
        remaining = set(range(len(candidate_idxs)))
        
        def score_candidate(local_idx: int) -> Tuple[float, Dict[str, float]]:
            """Compute composite score for candidate."""
            idx = candidate_idxs[local_idx]
            c = self.chunks[idx]
            cvec = self.chunk_embeddings[idx]
            
            # Component scores
            s_ret = float(ret_sims[local_idx])
            s_nov = self._novelty_score(cvec, selected_vecs)
            s_div = self._doc_diversity_boost(c["doc_id"], doc_counts, max_per_doc)
            
            total = w_ret * s_ret + w_nov * s_nov + w_div * s_div
            parts = {"retrieval": s_ret, "novelty": s_nov, "doc_diversity": s_div}
            return total, parts
        
        # Greedy selection loop
        while remaining and len(selected) < max_chunks:
            best = None
            best_score = -1e9
            best_parts = None
            
            # Find best-scoring candidate that passes constraints
            for li in list(remaining):
                idx = candidate_idxs[li]
                c = self.chunks[idx]
                
                # Token constraint
                ctoks = int(c["tokens"])
                if ctoks < self.policy["min_chunk_tokens"]:
                    remaining.remove(li)
                    excluded.append({"chunk_id": c["chunk_id"], "reason": "too_short"})
                    continue
                
                if used_tokens + ctoks > budget:
                    # Can't fit now; keep for later
                    continue
                
                # Per-document cap
                if doc_counts.get(c["doc_id"], 0) >= max_per_doc:
                    remaining.remove(li)
                    excluded.append({"chunk_id": c["chunk_id"], "reason": "doc_cap"})
                    continue
                
                # Redundancy check
                cvec = self.chunk_embeddings[idx]
                if self._is_redundant(cvec, selected_vecs, red_thr):
                    remaining.remove(li)
                    excluded.append({"chunk_id": c["chunk_id"], "reason": "redundant"})
                    continue
                
                # Score this candidate
                sc, parts = score_candidate(li)
                if sc > best_score:
                    best_score = sc
                    best = li
                    best_parts = parts
            
            # If nothing fits, stop
            if best is None:
                break
            
            # Accept best candidate
            idx = candidate_idxs[best]
            c = self.chunks[idx]
            ctoks = int(c["tokens"])
            
            selected.append({
                "chunk_id": c["chunk_id"],
                "doc_id": c["doc_id"],
                "title": c["title"],
                "heading": c.get("heading"),
                "tokens": ctoks,
                "score": float(best_score),
                "score_parts": best_parts,
                "source_path": c.get("source_path"),
                "text": c["text"]
            })
            
            used_tokens += ctoks
            doc_counts[c["doc_id"]] = doc_counts.get(c["doc_id"], 0) + 1
            selected_vecs = np.vstack([selected_vecs, self.chunk_embeddings[idx][None, :]])
            remaining.remove(best)
        
        # Mark remaining as not selected
        for li in sorted(list(remaining)):
            idx = candidate_idxs[li]
            excluded.append({"chunk_id": self.chunks[idx]["chunk_id"], "reason": "not_selected"})
        
        # Compute context-level metrics
        relevance = float(np.mean([s["score_parts"]["retrieval"] for s in selected])) if selected else 0.0
        
        redundancy = 0.0
        if len(selected) >= 2:
            sims = selected_vecs @ selected_vecs.T
            mask = ~np.eye(sims.shape[0], dtype=bool)
            redundancy = float(np.max(sims[mask]))
        
        diversity = float(len(set(s["doc_id"] for s in selected)) / max(1, len(selected)))
        budget_eff = float(used_tokens / budget) if budget > 0 else 0.0
        
        # Build context package
        context_package = {
            "query": query,
            "policy_version": self.policy["policy_version"],
            "policy": {k: self.policy[k] for k in self.policy},
            "budget_tokens": budget,
            "used_tokens": used_tokens,
            "selected": selected,
            "excluded": excluded,
            "composer_metrics": {
                "avg_retrieval_relevance": relevance,
                "worst_redundancy": redundancy,
                "doc_diversity_ratio": diversity,
                "budget_efficiency": budget_eff,
                "selected_count": len(selected)
            }
        }
        
        return context_package


def render_context_text(pkg: Dict[str, Any]) -> str:
    """
    Render context package as formatted text.
    
    Args:
        pkg: Context package from ContextComposer.compose()
        
    Returns:
        Formatted text representation of selected chunks
    """
    parts = []
    for i, s in enumerate(pkg["selected"], 1):
        header = f"[{i}] {s['title']} — {s.get('heading') or ''} (tokens={s['tokens']})"
        body = s["text"].strip()
        parts.append(header + "\n" + body)
    
    return "\n\n" + ("\n\n" + ("-" * 80) + "\n\n").join(parts)
