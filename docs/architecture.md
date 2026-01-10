# DCE Architecture

This document provides a deep dive into the architecture and design principles of the Dynamic Context Engine (DCE).

---

## System Overview

DCE implements a **multi-stage retrieval and composition pipeline** with autonomous policy evolution:

```
┌──────────────┐
│  Documents   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Chunking   │ ← Recursive text splitting
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Embedding   │ ← HuggingFace BGE-small
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Pinecone    │ ← Vector DB (long-term memory)
│   Storage    │
└──────┬───────┘
       │
       ▼ (Query time)
┌──────────────┐
│ High-Recall  │ ← Top-50 retrieval
│  Retrieval   │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────┐
│  Dynamic Context Composer        │
│  ┌────────────────────────────┐  │
│  │ Greedy Multi-Objective     │  │
│  │ Selection:                 │  │
│  │  • Relevance (0.55)        │  │
│  │  • Novelty (0.25)          │  │
│  │  • Doc Diversity (0.20)    │  │
│  └────────────────────────────┘  │
│  ┌────────────────────────────┐  │
│  │ Constraints:               │  │
│  │  • Budget tokens (2048)    │  │
│  │  • Max chunks (20)         │  │
│  │  • Redundancy filter (0.75)│  │
│  └────────────────────────────┘  │
└──────┬───────────────────────────┘
       │
       ▼
┌──────────────────────┐
│  Context Package     │
│  • Selected chunks   │
│  • Composition order │
│  • Metadata          │
│  • Metrics           │
└──────┬───────────────┘
       │
   ┌───┴────┐
   │        │
   ▼        ▼
┌────┐  ┌────────────┐
│LLM │  │ Evaluator  │
│Gen │  │ + Evolver  │
└────┘  └─────┬──────┘
              │
              ▼
        ┌──────────────┐
        │ Letta Agent  │
        │ (Autonomous) │
        └──────────────┘
```

---

## Core Components

### 1. Data Ingestion

**Module**: `src/ingestion/`

**Purpose**: Load raw documents from various sources.

**Key Classes**:
- `DataLoader`: Reads .txt, .md, .json, .jsonl files
- `PDFLoader`: Extracts text from PDFs (optional)
- `WebScraper`: Fetches content from URLs (optional)

**Implementation**:
```python
class DataLoader:
    def load_directory(self, path: str) -> List[Dict]:
        # Returns: [{"doc_id": str, "text": str, "metadata": dict}]
        pass
```

**Design Decisions**:
- Generic document format: `{"doc_id", "text", "metadata"}`
- Metadata preserved through pipeline
- Support for incremental loading

---

### 2. Chunking

**Module**: `src/chunking/`

**Purpose**: Split documents into semantically coherent chunks.

**Key Classes**:
- `DocumentChunker`: Recursive character-based splitting
- `SemanticChunker`: Embedding-based boundary detection (future)

**Algorithm**:
```python
def chunk_documents(docs, chunk_size=512, overlap=50):
    for doc in docs:
        # 1. Split on paragraph boundaries
        paragraphs = doc['text'].split('\n\n')
        
        # 2. Accumulate until chunk_size
        chunks = []
        current = ""
        for para in paragraphs:
            if len(current) + len(para) < chunk_size:
                current += para + "\n\n"
            else:
                chunks.append(current)
                current = para[-overlap:]  # Overlap
        
        # 3. Add metadata
        for i, chunk in enumerate(chunks):
            yield {
                "chunk_id": f"{doc['doc_id']}_chunk_{i}",
                "doc_id": doc['doc_id'],
                "text": chunk,
                "position": i,
                "metadata": doc['metadata']
            }
```

**Design Decisions**:
- **Chunk size**: 512 tokens (balance context vs granularity)
- **Overlap**: 50 tokens (preserve continuity)
- **Boundary detection**: Prefer natural breaks (paragraphs, sentences)

---

### 3. Embedding

**Module**: `src/embedding/`

**Purpose**: Convert text chunks to dense vector representations.

**Key Classes**:
- `HuggingFaceEmbedder`: API-based embedding
- `LocalEmbedder`: Local model inference (future)

**Model**: **BAAI/bge-small-en-v1.5**
- Dimension: 384
- Performance: MTEB score 62.17
- Speed: ~50ms per chunk (API)

**Implementation**:
```python
class HuggingFaceEmbedder:
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        # 1. Batch texts (max 32 per request)
        # 2. Call HuggingFace Inference API
        # 3. Normalize vectors (L2)
        # 4. Return: (N, 384) array
        pass
```

**Design Decisions**:
- **Normalization**: L2 norm for cosine similarity
- **Batching**: Reduce API calls
- **Caching**: Store embeddings in Pinecone (avoid re-generation)

---

### 4. Vector Database (Pinecone LTM)

**Module**: `src/vectordb/`

**Purpose**: Persistent storage and high-recall retrieval.

**Key Classes**:
- `PineconeLTM`: Pinecone client wrapper

**Operations**:
```python
class PineconeLTM:
    def upsert_chunks(self, chunks, embeddings, namespace):
        # Store chunks with embeddings
        pass
    
    def search(self, query: str, top_k=50, namespace=None):
        # Returns: (chunks, embeddings, query_vec)
        # - chunks: Top-k by cosine similarity
        # - embeddings: Pre-computed vectors (include_values=True)
        # - query_vec: Query embedding
        pass
```

**Index Configuration**:
- **Metric**: Cosine similarity
- **Dimension**: 384 (matches bge-small)
- **Namespaces**: Version control (v1, v2, experiment_xyz)
- **Metadata**: Full chunk content + metadata

**Why Pinecone?**
- Managed service (no infra overhead)
- Fast approximate NN search (<50ms for 1M vectors)
- Metadata filtering
- Built-in versioning (namespaces)

---

### 5. Dynamic Context Composer

**Module**: `src/context/`

**Purpose**: Select optimal subset of chunks via multi-objective optimization.

**Algorithm**: **Greedy Multi-Objective Selection**

```python
def compose(query, candidate_idxs, query_vec):
    selected = []
    remaining = set(candidate_idxs)
    used_tokens = 0
    
    while remaining and used_tokens < budget_tokens:
        # Score each remaining chunk
        scores = {}
        for idx in remaining:
            # 1. Retrieval score (semantic similarity)
            s_retrieval = cosine(query_vec, chunk_embeddings[idx])
            
            # 2. Novelty score (dissimilarity to selected)
            if selected:
                similarities = [cosine(chunk_embeddings[idx], chunk_embeddings[s]) 
                               for s in selected]
                s_novelty = 1 - max(similarities)
            else:
                s_novelty = 1.0
            
            # 3. Document diversity score
            doc_id = chunks[idx]['doc_id']
            doc_count = sum(1 for s in selected if chunks[s]['doc_id'] == doc_id)
            if doc_count >= max_chunks_per_doc:
                s_doc_div = 0.0  # Penalize over-represented docs
            else:
                s_doc_div = 1.0 / (1 + doc_count)
            
            # 4. Redundancy check
            if selected and max(similarities) > redundancy_threshold:
                s_redundancy_penalty = redundancy_penalty
            else:
                s_redundancy_penalty = 0.0
            
            # Combined score
            scores[idx] = (
                w_retrieval * s_retrieval +
                w_novelty * s_novelty +
                w_doc_diversity * s_doc_div -
                s_redundancy_penalty
            )
        
        # Select best chunk
        best_idx = max(scores, key=scores.get)
        selected.append(best_idx)
        remaining.remove(best_idx)
        used_tokens += estimate_tokens(chunks[best_idx]['text'])
    
    return {
        "selected": [chunks[i] for i in selected],
        "selected_idxs": selected,
        "used_tokens": used_tokens,
        "metrics": compute_metrics(selected)
    }
```

**Key Features**:
1. **Greedy selection**: Iteratively pick best chunk
2. **Multi-objective**: Balance relevance, novelty, diversity
3. **Constraints**: Token budget, max chunks, max per doc
4. **Redundancy filtering**: Skip similar chunks (cosine > 0.75)

**Design Decisions**:
- **Greedy vs Optimal**: Greedy is O(n²), optimal is NP-hard
- **Weights**: Tuned via policy evolution (baseline: 0.55/0.25/0.20)
- **Budget-aware**: Stop when token limit reached

---

### 6. Context Evaluator

**Module**: `src/evaluation/`

**Purpose**: Compute objective metrics for composition quality.

**Metrics**:

```python
def evaluate(context_pkg):
    selected = context_pkg['selected']
    
    # 1. Relevance: Avg similarity to query
    relevance = np.mean([cosine(query_vec, chunk_embeddings[i]) 
                        for i in selected_idxs])
    
    # 2. Redundancy: Avg pairwise similarity (lower is better)
    pairs = [(i, j) for i in selected_idxs for j in selected_idxs if i < j]
    redundancy = np.mean([cosine(chunk_embeddings[i], chunk_embeddings[j]) 
                         for i, j in pairs])
    
    # 3. Coverage: Unique documents
    unique_docs = len(set(c['doc_id'] for c in selected))
    coverage = unique_docs / total_docs_available
    
    # 4. Efficiency: Budget utilization
    efficiency = used_tokens / budget_tokens
    
    # 5. Overall: Weighted combination
    overall = (
        0.35 * relevance +
        0.25 * (1 - redundancy) +  # Inverse redundancy
        0.20 * coverage +
        0.20 * efficiency
    )
    
    return {
        "relevance": relevance,
        "redundancy": redundancy,
        "coverage": coverage,
        "efficiency": efficiency,
        "overall": overall
    }
```

**Why These Metrics?**
- **No LLM judge**: Objective, fast, reproducible
- **Retrieval quality**: Relevance measures semantic match
- **Information density**: Low redundancy = more unique info
- **Breadth**: Coverage measures topic diversity
- **Resource usage**: Efficiency prevents over/under-filling

---

### 7. Policy Evolution

**Module**: `src/evolution/`

**Purpose**: Automatically discover better composition policies.

**Algorithm**: **Experiment-Based Evolution**

```python
def evolve_policy(baseline_policy, baseline_contexts, queries):
    current_policy = baseline_policy
    current_score = evaluate_policy(current_policy, queries)
    
    for iteration in range(max_iterations):
        # 1. Propose variation
        candidate_policy = perturb_policy(current_policy)
        
        # 2. Evaluate on same queries
        candidate_score = evaluate_policy(candidate_policy, queries)
        
        # 3. Compare
        if candidate_score > current_score:
            current_policy = candidate_policy
            current_score = candidate_score
            print(f"Iteration {iteration}: Improved to {current_score:.3f}")
        
        # 4. Log results
        log_experiment(iteration, candidate_policy, candidate_score)
    
    return current_policy
```

**Perturbation Strategies**:
- **Weight adjustment**: ±0.05 on composition weights
- **Threshold tuning**: ±0.05 on redundancy_threshold
- **Constraint relaxation**: ±2 tokens on budget

**Stopping Criteria**:
- Max iterations reached
- No improvement for 5 iterations
- Score plateau (<1% change)

---

### 8. Letta Agents (Autonomous)

**Module**: `src/letta/`

**Purpose**: Autonomous policy proposal and decision-making.

**Agents**:

#### 8.1 **Policy Strategist**
```python
agent = create_policy_strategist()

# Propose experiment
proposal = agent.send_message(
    f"Baseline metrics: {baseline_metrics}\n"
    f"Current policy: {current_policy}\n"
    f"Propose a new policy to improve diversity."
)
# Returns: {"new_policy": {...}, "rationale": "..."}
```

**Capabilities**:
- Analyze baseline performance
- Identify bottlenecks (low diversity, high redundancy)
- Propose targeted changes
- Estimate impact

#### 8.2 **Experiment Decider**
```python
decision = agent.send_message(
    f"Baseline: {baseline_metrics}\n"
    f"Candidate: {candidate_metrics}\n"
    f"Decide: ACCEPT or REJECT?"
)
# Returns: {"decision": "ACCEPT", "reason": "..."}
```

**Capabilities**:
- Compare statistical significance
- Balance trade-offs (diversity vs relevance)
- Reject marginal improvements
- Approve only robust gains

**Memory**:
- **Core Memory**: Current policy, past experiments
- **Archival Memory**: Full experiment history (queryable)
- **Recall Memory**: Recent conversation context

**Why Letta?**
- Stateful agents (remember past experiments)
- Long-term memory (archival storage)
- Human-in-the-loop option
- Tool calling (can trigger experiments)

---

### 9. Self-Evolving Loop

**Module**: `src/orchestration/`

**Purpose**: Fully automated evolution cycle.

**Workflow**:
```python
loop = SelfEvolvingLoop(chunks, embeddings, letta_agent)

loop.run_evolution_cycle(
    queries=test_queries,
    max_iterations=10
)
```

**Steps**:
1. **Baseline Evaluation**: Run current policy on test queries
2. **Agent Proposal**: Letta proposes new policy
3. **Experiment**: Evaluate new policy on same queries
4. **Agent Decision**: Letta compares and decides
5. **Update**: If accepted, set as new baseline
6. **Repeat**: Goto step 2

**Stopping Criteria**:
- Max iterations
- Agent decides "sufficient performance"
- No proposals from agent

---

## Design Principles

### 1. **Modularity**
- Each component is independently usable
- Clear interfaces (input/output contracts)
- Easy to swap implementations (e.g., different embedders)

### 2. **Reproducibility**
- Objective metrics (no LLM judges)
- Seed control for random operations
- Experiment logging

### 3. **Scalability**
- Batch processing for embeddings
- Pinecone handles 1M+ vectors
- Greedy algorithm is O(n²), acceptable for n=50

### 4. **Extensibility**
- Policy parameters are configurable
- Evaluation weights are tunable
- Easy to add new metrics or constraints

### 5. **Production-Ready**
- Error handling and retries
- Comprehensive logging
- Unit tests for core functions
- API compatibility (LangChain, LlamaIndex)

---

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Embedding | O(n) | HF API: ~50ms/chunk |
| Pinecone Upsert | O(n log n) | Serverless index |
| Pinecone Search | O(log n) | ANN search |
| Composition | O(k²) | k = selected chunks (~10-20) |
| Evaluation | O(k²) | Pairwise similarities |

### Space Complexity

| Component | Space | Notes |
|-----------|-------|-------|
| Embeddings | O(n × d) | n chunks, d=384 |
| Candidates | O(k) | k=50 candidates |
| Selected | O(m) | m=10-20 selected |

### Latency Benchmarks

| Operation | Time | Dataset |
|-----------|------|---------|
| Full Pipeline | ~1s | 5,662 chunks |
| Pinecone Search | 50ms | 1M vectors |
| Composition | 30ms | 50 candidates |
| Evaluation | 10ms | 20 selected |

---

## Trade-offs

### Greedy vs Optimal Selection
- **Greedy**: Fast (O(n²)), good results
- **Optimal**: Slow (NP-hard), marginal improvement
- **Choice**: Greedy (pragmatic)

### Redundancy Threshold
- **High (0.85)**: Very strict, may under-select
- **Medium (0.75)**: Balanced (current)
- **Low (0.65)**: Permissive, more redundancy
- **Choice**: 0.75 (tested on biotech dataset)

### Embedding Model
- **Large (1024d)**: Better accuracy, slower, expensive
- **Small (384d)**: Fast, cheap, sufficient for most tasks
- **Choice**: bge-small-en-v1.5 (384d)

---

## Future Enhancements

1. **Streaming Composition**: Real-time chunk selection
2. **Multi-Modal**: Images, tables, code
3. **Semantic Chunking**: Embedding-based boundaries
4. **Local Embeddings**: On-device inference
5. **Advanced Optimization**: Genetic algorithms, RL

---

**Next**: [API Reference →](api-reference.md)
