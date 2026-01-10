# Getting Started with DCE

This guide will walk you through installing DCE, setting up your environment, and running your first dynamic context composition.

---

## Prerequisites

- **Python 3.9+** (3.10 recommended)
- **pip** or **conda**
- API keys for:
  - **Letta** (https://app.letta.com) - for autonomous agents
  - **Pinecone** (https://pinecone.io) - for vector database
  - **HuggingFace** (https://huggingface.co) - for embeddings

---

## Installation

### Option 1: Standard Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/self-evolving-dce.git
cd self-evolving-dce

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Package Installation

```bash
pip install -e .
```

### Option 3: With Development Tools

```bash
pip install -e ".[dev,docs]"
```

---

## Configuration

### 1. Environment Variables

Create a `.env` file in the project root:

```bash
# Copy the template
cp .env.example .env
```

Edit `.env` with your credentials:

```ini
# Letta Configuration
LETTA_API_KEY=letta_sk_...

# Pinecone Configuration
PINECONE_API_KEY=pcsk_...
PINECONE_INDEX_NAME=dce-biotech

# HuggingFace Configuration
HF_TOKEN=hf_...

# Optional: Model Configuration
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
LETTA_MODEL=gpt-4o-mini
```

### 2. Pinecone Index Setup

```python
from pinecone import Pinecone, ServerlessSpec

# Initialize Pinecone
pc = Pinecone(api_key="your-key")

# Create index (one-time setup)
pc.create_index(
    name="dce-biotech",
    dimension=384,  # For bge-small-en-v1.5
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)
```

### 3. Verify Installation

```python
from src import ContextComposer, PineconeLTM, PolicyEvolver

print("✅ All imports successful!")
```

---

## Your First Query

### Step 1: Load Data

```python
from src import DataLoader

loader = DataLoader()
documents = loader.load_directory("data/")

print(f"Loaded {len(documents)} documents")
```

### Step 2: Chunk Documents

```python
from src import DocumentChunker

chunker = DocumentChunker(
    chunk_size=512,
    chunk_overlap=50,
    chunk_method="recursive"
)

chunks = chunker.chunk_documents(documents)
print(f"Created {len(chunks)} chunks")
```

### Step 3: Generate Embeddings

```python
from src import HuggingFaceEmbedder

embedder = HuggingFaceEmbedder(
    model="BAAI/bge-small-en-v1.5",
    api_key="hf_..."
)

embeddings = embedder.embed_texts([c['text'] for c in chunks])
print(f"Generated {len(embeddings)} embeddings")
```

### Step 4: Upload to Pinecone

```python
from src import PineconeLTM

vectordb = PineconeLTM(
    api_key="pcsk_...",
    index_name="dce-biotech"
)

vectordb.upsert_chunks(chunks, embeddings, namespace="v1")
print("✅ Data uploaded to Pinecone")
```

### Step 5: Query with Dynamic Composition

```python
# Define your query
query = "What is CRISPR gene editing?"

# High-recall retrieval (top-50)
candidates, candidate_embeddings, query_vec = vectordb.search(
    query=query,
    top_k=50,
    namespace="v1"
)

# Dynamic context composition
from src import ContextComposer

composer = ContextComposer(
    chunks=candidates,
    chunk_embeddings=candidate_embeddings,
    policy={
        "w_retrieval": 0.55,
        "w_novelty": 0.25,
        "w_doc_diversity": 0.20,
        "budget_tokens": 2048,
        "redundancy_threshold": 0.75
    }
)

context_pkg = composer.compose(
    query=query,
    candidate_idxs=list(range(len(candidates))),
    query_vec=query_vec
)

# Results
print(f"Selected chunks: {len(context_pkg['selected'])}")
print(f"Used tokens: {context_pkg['used_tokens']}")
print(f"Coverage: {context_pkg['metrics']['coverage']:.2%}")
```

### Step 6: Use Context with LLM

```python
import openai

# Combine selected chunks
context_text = "\n\n".join([
    f"[Doc {c['doc_id']}] {c['text']}"
    for c in context_pkg['selected']
])

# Generate response
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a biotech expert."},
        {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
    ]
)

print(response.choices[0].message.content)
```

---

## Understanding the Policy

The policy dictionary controls composition behavior:

```python
policy = {
    # === Composition Weights ===
    "w_retrieval": 0.55,        # Semantic similarity to query
    "w_novelty": 0.25,          # Dissimilarity to already-selected
    "w_doc_diversity": 0.20,    # Spread across source documents
    
    # === Constraints ===
    "budget_tokens": 2048,      # Maximum context window
    "max_chunks": 20,           # Hard limit on chunk count
    "max_chunks_per_doc": 3,    # Enforce diversity
    
    # === Redundancy Control ===
    "redundancy_threshold": 0.75,  # Cosine similarity cutoff
    "redundancy_penalty": 0.3,     # Score penalty for similar chunks
}
```

**Key Tuning Parameters:**

- **Higher `w_retrieval`**: More relevance, less diversity
- **Higher `w_novelty`**: More unique information per chunk
- **Higher `w_doc_diversity`**: Broader source coverage
- **Lower `redundancy_threshold`**: Stricter filtering (0.8 = very strict)
- **Higher `budget_tokens`**: Larger context window

---

## Evaluation Metrics

Evaluate composition quality:

```python
from src import ContextEvaluator

evaluator = ContextEvaluator()
metrics = evaluator.evaluate(context_pkg)

print(f"Relevance: {metrics['relevance']:.2%}")
print(f"Redundancy: {metrics['redundancy']:.2%}")  # Lower is better
print(f"Coverage: {metrics['coverage']:.2%}")
print(f"Efficiency: {metrics['efficiency']:.2%}")
print(f"Overall Score: {metrics['overall']:.2%}")
```

**Metric Interpretation:**

- **Relevance** (0-1): Average similarity to query
- **Redundancy** (0-1): Average pairwise chunk similarity (↓ is better)
- **Coverage** (0-1): Unique content across documents
- **Efficiency** (0-1): Budget utilization (0.8-0.95 is ideal)
- **Overall** (0-1): Weighted combination

---

## Next Steps

1. **Explore Examples**
   - [`examples/full_pipeline.py`](../examples/full_pipeline.py)
   - [`examples/custom_integration.py`](../examples/custom_integration.py)

2. **Read Architecture**
   - [Architecture Deep Dive](architecture.md)

3. **Policy Evolution**
   - [Letta Agents Documentation](letta-agents.md)

4. **Integration**
   - [Integration Guide](integration-guide.md) (LangChain, LlamaIndex, FastAPI)

5. **Production Deployment**
   - [Production Guide](production-deployment.md)

---

## Troubleshooting

### Error: "Pinecone index not found"
```bash
# Create the index first
python scripts/setup_pinecone.py
```

### Error: "HuggingFace API 500"
```python
# Add retry logic
embedder = HuggingFaceEmbedder(max_retries=3, retry_delay=2)
```

### Error: "Out of memory"
```python
# Process in batches
for i in range(0, len(chunks), 100):
    batch = chunks[i:i+100]
    embeddings = embedder.embed_texts(batch)
    vectordb.upsert_chunks(batch, embeddings)
```

### Error: "Letta agent timeout"
```python
# Increase timeout
agent = create_policy_strategist(timeout=60)
```

---

## Getting Help

- **Documentation**: [docs/](.)
- **Examples**: [examples/](../examples/)
- **Issues**: [GitHub Issues](https://github.com/yourusername/self-evolving-dce/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/self-evolving-dce/discussions)

---

**Next**: [Architecture Deep Dive →](architecture.md)
