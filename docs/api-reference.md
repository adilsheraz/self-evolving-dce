# API Reference

Complete API documentation for the Dynamic Context Engine.

---

## Table of Contents

- [Data Ingestion](#data-ingestion)
- [Chunking](#chunking)
- [Embedding](#embedding)
- [Vector Database](#vector-database)
- [Context Composition](#context-composition)
- [Evaluation](#evaluation)
- [Policy Evolution](#policy-evolution)
- [Letta Agents](#letta-agents)
- [Orchestration](#orchestration)

---

## Data Ingestion

### `DataLoader`

Load documents from various sources.

```python
from src import DataLoader

loader = DataLoader()
```

#### Methods

##### `load_directory(path: str) -> List[Dict]`

Load all documents from a directory.

**Parameters:**
- `path` (str): Directory path

**Returns:**
- List[Dict]: Documents with structure `{"doc_id": str, "text": str, "metadata": dict}`

**Example:**
```python
docs = loader.load_directory("data/biotech")
# [{"doc_id": "doc_001", "text": "...", "metadata": {...}}, ...]
```

---

## Chunking

### `DocumentChunker`

Split documents into chunks.

```python
from src import DocumentChunker

chunker = DocumentChunker(
    chunk_size=512,
    chunk_overlap=50,
    chunk_method="recursive"
)
```

#### Parameters

- `chunk_size` (int, default=512): Target chunk size in characters
- `chunk_overlap` (int, default=50): Overlap between chunks
- `chunk_method` (str, default="recursive"): Chunking strategy

#### Methods

##### `chunk_documents(documents: List[Dict]) -> List[Dict]`

Chunk a list of documents.

**Parameters:**
- `documents` (List[Dict]): Documents from DataLoader

**Returns:**
- List[Dict]: Chunks with structure:
  ```python
  {
      "chunk_id": "doc_001_chunk_0",
      "doc_id": "doc_001",
      "text": "chunk text...",
      "position": 0,
      "metadata": {...}
  }
  ```

**Example:**
```python
chunks = chunker.chunk_documents(docs)
print(f"Created {len(chunks)} chunks")
```

---

## Embedding

### `HuggingFaceEmbedder`

Generate embeddings via HuggingFace Inference API.

```python
from src import HuggingFaceEmbedder

embedder = HuggingFaceEmbedder(
    model="BAAI/bge-small-en-v1.5",
    api_key="hf_...",
    max_retries=3,
    retry_delay=2
)
```

#### Parameters

- `model` (str): HuggingFace model ID
- `api_key` (str): HF API token
- `max_retries` (int, default=3): Retry on API failure
- `retry_delay` (int, default=2): Delay between retries (seconds)

#### Methods

##### `embed_texts(texts: List[str]) -> np.ndarray`

Embed a list of texts.

**Parameters:**
- `texts` (List[str]): Text strings to embed

**Returns:**
- np.ndarray: Shape (N, D) where D is embedding dimension (384 for bge-small)

**Example:**
```python
texts = [c['text'] for c in chunks]
embeddings = embedder.embed_texts(texts)
# Shape: (5662, 384)
```

---

## Vector Database

### `PineconeLTM`

Pinecone vector database client.

```python
from src import PineconeLTM

vectordb = PineconeLTM(
    api_key="pcsk_...",
    index_name="dce-biotech",
    cloud="aws",
    region="us-east-1"
)
```

#### Parameters

- `api_key` (str): Pinecone API key
- `index_name` (str): Index name
- `cloud` (str, optional): Cloud provider
- `region` (str, optional): Region

#### Methods

##### `upsert_chunks(chunks: List[Dict], embeddings: np.ndarray, namespace: str = None) -> None`

Upload chunks and embeddings to Pinecone.

**Parameters:**
- `chunks` (List[Dict]): Chunks from DocumentChunker
- `embeddings` (np.ndarray): Embeddings from HuggingFaceEmbedder
- `namespace` (str, optional): Namespace for versioning

**Example:**
```python
vectordb.upsert_chunks(chunks, embeddings, namespace="v1")
```

##### `search(query: str, top_k: int = 50, namespace: str = None) -> Tuple`

Search for similar chunks.

**Parameters:**
- `query` (str): Query text
- `top_k` (int, default=50): Number of results
- `namespace` (str, optional): Namespace to search

**Returns:**
- Tuple: `(chunks, embeddings, query_vec)`
  - `chunks` (List[Dict]): Top-k chunks
  - `embeddings` (np.ndarray): Chunk embeddings (shape: k×384)
  - `query_vec` (np.ndarray): Query embedding (shape: 384)

**Example:**
```python
candidates, candidate_embeddings, query_vec = vectordb.search(
    query="What is CRISPR?",
    top_k=50,
    namespace="v1"
)
```

---

## Context Composition

### `ContextComposer`

Dynamic multi-objective context composition.

```python
from src import ContextComposer

composer = ContextComposer(
    chunks=candidates,
    chunk_embeddings=candidate_embeddings,
    policy={
        "w_retrieval": 0.55,
        "w_novelty": 0.25,
        "w_doc_diversity": 0.20,
        "budget_tokens": 2048,
        "max_chunks": 20,
        "max_chunks_per_doc": 3,
        "redundancy_threshold": 0.75,
        "redundancy_penalty": 0.3
    }
)
```

#### Parameters

- `chunks` (List[Dict]): Candidate chunks
- `chunk_embeddings` (np.ndarray): Chunk embeddings
- `policy` (Dict): Composition policy (see below)

#### Policy Parameters

```python
{
    # Composition weights (should sum to ~1.0)
    "w_retrieval": float,       # Weight for semantic similarity
    "w_novelty": float,          # Weight for dissimilarity to selected
    "w_doc_diversity": float,    # Weight for document spread
    
    # Constraints
    "budget_tokens": int,        # Max context window size
    "max_chunks": int,           # Hard limit on chunk count
    "max_chunks_per_doc": int,   # Max chunks from same document
    
    # Redundancy control
    "redundancy_threshold": float,  # Cosine similarity cutoff (0-1)
    "redundancy_penalty": float,    # Score penalty for similar chunks
}
```

#### Methods

##### `compose(query: str, candidate_idxs: List[int], query_vec: np.ndarray) -> Dict`

Compose optimal context package.

**Parameters:**
- `query` (str): Query text
- `candidate_idxs` (List[int]): Indices of candidate chunks
- `query_vec` (np.ndarray): Query embedding

**Returns:**
- Dict: Context package with structure:
  ```python
  {
      "selected": List[Dict],      # Selected chunks
      "selected_idxs": List[int],  # Indices of selected
      "used_tokens": int,           # Total tokens used
      "metrics": Dict               # Composition metrics
  }
  ```

**Example:**
```python
context_pkg = composer.compose(
    query="What is CRISPR?",
    candidate_idxs=list(range(len(candidates))),
    query_vec=query_vec
)

print(f"Selected {len(context_pkg['selected'])} chunks")
print(f"Used {context_pkg['used_tokens']} tokens")
```

---

## Evaluation

### `ContextEvaluator`

Evaluate composition quality with objective metrics.

```python
from src import ContextEvaluator

evaluator = ContextEvaluator(
    weights={
        "relevance": 0.35,
        "redundancy": 0.25,
        "coverage": 0.20,
        "efficiency": 0.20
    }
)
```

#### Parameters

- `weights` (Dict, optional): Metric weights for overall score

#### Methods

##### `evaluate(context_pkg: Dict) -> Dict`

Compute evaluation metrics.

**Parameters:**
- `context_pkg` (Dict): Output from ContextComposer.compose()

**Returns:**
- Dict: Metrics with structure:
  ```python
  {
      "relevance": float,     # Avg similarity to query (0-1)
      "redundancy": float,    # Avg pairwise similarity (0-1, lower is better)
      "coverage": float,      # Unique document ratio (0-1)
      "efficiency": float,    # Budget utilization (0-1)
      "overall": float        # Weighted combination (0-1)
  }
  ```

**Example:**
```python
metrics = evaluator.evaluate(context_pkg)
print(f"Relevance: {metrics['relevance']:.2%}")
print(f"Redundancy: {metrics['redundancy']:.2%}")
print(f"Overall Score: {metrics['overall']:.2%}")
```

---

## Policy Evolution

### `PolicyEvolver`

Automatically discover better composition policies.

```python
from src import PolicyEvolver

evolver = PolicyEvolver(
    baseline_policy=current_policy,
    baseline_contexts=baseline_contexts,
    max_iterations=10,
    patience=5
)
```

#### Parameters

- `baseline_policy` (Dict): Starting policy
- `baseline_contexts` (List[Dict]): Baseline context packages
- `max_iterations` (int, default=10): Max evolution iterations
- `patience` (int, default=5): Stop if no improvement for N iterations

#### Methods

##### `evolve(queries: List[str]) -> Dict`

Run policy evolution.

**Parameters:**
- `queries` (List[str]): Test queries for evaluation

**Returns:**
- Dict: Best policy found

**Example:**
```python
test_queries = [
    "What is CRISPR?",
    "Explain de-extinction",
    # ...
]

best_policy = evolver.evolve(queries=test_queries)
print(f"New policy: {best_policy}")
```

---

## Letta Agents

### `create_policy_strategist()`

Create Letta agent for policy proposals.

```python
from src.letta import create_policy_strategist

agent = create_policy_strategist(
    api_key="letta_sk_...",
    model="gpt-4o-mini",
    timeout=30
)
```

#### Parameters

- `api_key` (str, optional): Letta API key (or from env)
- `model` (str, default="gpt-4o-mini"): LLM model
- `timeout` (int, default=30): Request timeout

#### Methods

##### `send_message(message: str) -> Dict`

Send message to agent.

**Parameters:**
- `message` (str): Prompt for agent

**Returns:**
- Dict: Agent response (parsed JSON)

**Example:**
```python
proposal = agent.send_message(
    f"Baseline metrics: {baseline_metrics}\n"
    f"Propose a new policy to improve diversity."
)
# {"new_policy": {...}, "rationale": "..."}
```

### `create_experiment_decider()`

Create Letta agent for experiment decisions.

```python
from src.letta import create_experiment_decider

decider = create_experiment_decider()
```

**Usage:**
```python
decision = decider.send_message(
    f"Baseline: {baseline_metrics}\n"
    f"Candidate: {candidate_metrics}\n"
    f"Decision: ACCEPT or REJECT?"
)
# {"decision": "ACCEPT", "reason": "..."}
```

---

## Orchestration

### `SelfEvolvingLoop`

Fully automated evolution cycle with Letta agents.

```python
from src.orchestration import SelfEvolvingLoop

loop = SelfEvolvingLoop(
    chunks=chunks,
    chunk_embeddings=embeddings,
    letta_agent=agent,
    max_iterations=10
)
```

#### Parameters

- `chunks` (List[Dict]): All chunks
- `chunk_embeddings` (np.ndarray): All embeddings
- `letta_agent`: Letta policy strategist agent
- `max_iterations` (int, default=10): Max evolution cycles

#### Methods

##### `run_evolution_cycle(queries: List[str]) -> Dict`

Run full evolution cycle.

**Parameters:**
- `queries` (List[str]): Test queries

**Returns:**
- Dict: Final policy and results

**Example:**
```python
results = loop.run_evolution_cycle(queries=test_queries)
print(f"Final policy: {results['final_policy']}")
print(f"Improvement: +{results['improvement_pct']:.1f}%")
```

---

## Utility Functions

### `estimate_tokens(text: str) -> int`

Estimate token count for text.

```python
from src import estimate_tokens

tokens = estimate_tokens("This is a sample text.")
# Returns: approximate token count
```

### `cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float`

Compute cosine similarity between two vectors.

```python
from src import cosine_similarity

sim = cosine_similarity(query_vec, chunk_vec)
# Returns: similarity in range [0, 1]
```

---

## Type Definitions

### Document
```python
{
    "doc_id": str,
    "text": str,
    "metadata": dict
}
```

### Chunk
```python
{
    "chunk_id": str,
    "doc_id": str,
    "text": str,
    "position": int,
    "metadata": dict
}
```

### Context Package
```python
{
    "selected": List[Chunk],
    "selected_idxs": List[int],
    "used_tokens": int,
    "metrics": {
        "relevance": float,
        "redundancy": float,
        "coverage": float,
        "efficiency": float,
        "overall": float
    }
}
```

### Policy
```python
{
    "w_retrieval": float,
    "w_novelty": float,
    "w_doc_diversity": float,
    "budget_tokens": int,
    "max_chunks": int,
    "max_chunks_per_doc": int,
    "redundancy_threshold": float,
    "redundancy_penalty": float
}
```

---

## Constants

### Default Policy
```python
DEFAULT_POLICY = {
    "w_retrieval": 0.55,
    "w_novelty": 0.25,
    "w_doc_diversity": 0.20,
    "budget_tokens": 2048,
    "max_chunks": 20,
    "max_chunks_per_doc": 3,
    "redundancy_threshold": 0.75,
    "redundancy_penalty": 0.3
}
```

### Evaluation Weights
```python
EVAL_WEIGHTS = {
    "relevance": 0.35,
    "redundancy": 0.25,
    "coverage": 0.20,
    "efficiency": 0.20
}
```

---

## Error Handling

All methods may raise the following exceptions:

- `ValueError`: Invalid parameters
- `ConnectionError`: API connection failure
- `TimeoutError`: Request timeout
- `KeyError`: Missing required fields

**Example:**
```python
try:
    embeddings = embedder.embed_texts(texts)
except ConnectionError as e:
    print(f"HuggingFace API error: {e}")
    # Implement retry logic
```

---

**Next**: [Integration Guide →](integration-guide.md)
