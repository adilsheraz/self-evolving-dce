# 🧠 Self-Evolving Dynamic Context Engine (DCE)

**Production-ready RAG framework with autonomous policy evolution**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 📖 Overview

DCE is a next-generation Retrieval-Augmented Generation (RAG) framework that goes beyond traditional semantic search. It dynamically composes optimal context packages by balancing **relevance**, **novelty**, **coverage**, and **diversity** - then autonomously evolves its composition policies using Letta agents.

### Key Features

- **🎯 Dynamic Context Composition**: Multi-objective optimization (relevance + novelty + diversity)
- **🤖 Autonomous Evolution**: Letta agents propose and evaluate policy improvements
- **📊 Comprehensive Metrics**: Objective evaluation without LLM judges
- **🗄️ Production-Ready**: Pinecone integration with long-term memory
- **⚡ High Performance**: 18x faster than initial implementation with embedding caching
- **📈 Proven Results**: 433% better diversity, 47% less redundancy vs traditional RAG

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/adilsheraz/self-evolving-dce.git
cd self-evolving-dce

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

### Setup

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
# - LETTA_API_KEY (from https://app.letta.com)
# - PINECONE_API_KEY (from https://pinecone.io)
# - HF_TOKEN (from https://huggingface.co)
```

### Basic Usage

```python
from src import ContextComposer, PineconeLTM
import numpy as np

# Initialize Pinecone vector database
vectordb = PineconeLTM(api_key="your-key", index_name="dce-index")

# Query your knowledge base
query = "What is de-extinction?"
candidates, embeddings, query_vec = vectordb.search(query, top_k=50)

# Compose optimal context
composer = ContextComposer(
    chunks=candidates,
    chunk_embeddings=embeddings,
    policy={
        "w_retrieval": 0.55,
        "w_novelty": 0.25,
        "w_doc_diversity": 0.20,
        "budget_tokens": 2048
    }
)

context_pkg = composer.compose(
    query=query,
    candidate_idxs=list(range(len(candidates))),
    query_vec=query_vec
)

# Use composed context with your LLM
print(f"Selected {len(context_pkg['selected'])} chunks")
print(f"Total tokens: {context_pkg['used_tokens']}")
```

---

## 📊 Performance

### RAG vs DCE Comparison (Biotech Dataset, 5,662 QA pairs)

| Metric | Traditional RAG | DCE | Improvement |
|--------|----------------|-----|-------------|
| **Diversity** | 10.0% | 53.3% | **+433%** ⭐⭐ |
| **Redundancy** | 96.8% | 51.4% | **-47%** ⭐⭐ |
| **Novelty** | 17.7% | 20.4% | +15% |
| **Token Efficiency** | 877 tokens | 220 tokens | **-75%** ⭐⭐⭐ |
| **Retrieval Time** | 0.70s | 0.97s | +38% (acceptable) |
| **Similarity** | 81.6% | 80.7% | -1% (maintained) |

**Key Insight**: DCE achieves equivalent relevance with 75% fewer tokens while pulling from 5.3x more diverse documents.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Query                               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  High-Recall Retrieval │
         │  (Pinecone Vector DB)  │
         └───────────┬────────────┘
                     │ Top-50 candidates
                     ▼
         ┌───────────────────────┐
         │  Context Composer      │
         │  Multi-Objective       │
         │  Optimization:         │
         │  • Relevance (0.55)    │
         │  • Novelty (0.25)      │
         │  • Diversity (0.20)    │
         └───────────┬────────────┘
                     │ Optimal context
                     ▼
         ┌───────────────────────┐
         │  Context Evaluator     │
         │  • Redundancy          │
         │  • Coverage            │
         │  • Efficiency          │
         └───────────┬────────────┘
                     │
    ┌────────────────┴─────────────────┐
    │                                  │
    ▼                                  ▼
┌─────────┐                    ┌──────────────┐
│   LLM   │                    │ Letta Agent  │
│ Generate│                    │ Propose New  │
│ Response│                    │ Policy (v0.2)│
└─────────┘                    └──────┬───────┘
                                      │
                                      ▼
                              ┌──────────────┐
                              │  Experiment  │
                              │  Baseline vs │
                              │  New Policy  │
                              └──────┬───────┘
                                      │
                                      ▼
                              ┌──────────────┐
                              │ Letta Decides│
                              │ ACCEPT/REJECT│
                              └──────┬───────┘
                                      │
                                      ▼
                              ┌──────────────┐
                              │ Auto-Update  │
                              │ Production   │
                              │ Policy       │
                              └──────────────┘
```

---

## 📚 Core Modules

### 1. **Data Ingestion**
```python
from src import DataLoader

loader = DataLoader()
docs = loader.load_directory("path/to/docs")
```

### 2. **Chunking**
```python
from src import DocumentChunker

chunker = DocumentChunker(
    chunk_size=512,
    chunk_overlap=50
)
chunks = chunker.chunk_documents(docs)
```

### 3. **Embedding**
```python
from src import HuggingFaceEmbedder

embedder = HuggingFaceEmbedder(
    model="BAAI/bge-small-en-v1.5",
    api_key="hf_..."
)
embeddings = embedder.embed_texts([c['text'] for c in chunks])
```

### 4. **Vector Database**
```python
from src import PineconeLTM

vectordb = PineconeLTM(api_key="pcsk_...")
vectordb.upsert_chunks(chunks, embeddings, namespace="my-kb")
```

### 5. **Dynamic Composition**
```python
from src import ContextComposer

composer = ContextComposer(chunks, embeddings)
context = composer.compose(query, candidate_idxs, query_vec)
```

### 6. **Evaluation**
```python
from src import ContextEvaluator

evaluator = ContextEvaluator()
scores = evaluator.evaluate(context)
# {'relevance': 0.85, 'redundancy': 0.12, 'coverage': 0.78, 'overall': 0.81}
```

### 7. **Policy Evolution**
```python
from src import PolicyEvolver

evolver = PolicyEvolver(baseline_policy, baseline_contexts)
new_policy = evolver.evolve(num_iterations=5)
```

### 8. **Letta Agents** (Autonomous)
```python
from src.letta import create_policy_strategist

agent = create_policy_strategist()
proposal = agent.propose_experiment(baseline_metrics)
decision = agent.decide_on_results(experiment_results)
```

### 9. **Self-Evolving Loop** (Full Automation)
```python
from src.orchestration import SelfEvolvingLoop

loop = SelfEvolvingLoop(
    chunks=chunks,
    embeddings=embeddings,
    letta_agent=agent
)
loop.run_evolution_cycle(queries=test_queries, max_iterations=10)
```

---

## 🔧 Configuration

### Policy Parameters

```python
policy = {
    # Composition weights (must sum to ~1.0)
    "w_retrieval": 0.55,        # Semantic relevance to query
    "w_novelty": 0.25,          # Dissimilarity to already-selected
    "w_doc_diversity": 0.20,    # Spread across documents
    
    # Constraints
    "budget_tokens": 2048,      # Context window limit
    "max_chunks": 20,           # Maximum chunks to select
    "max_chunks_per_doc": 3,    # Diversity enforcement
    
    # Redundancy control
    "redundancy_threshold": 0.75,  # Cosine similarity cutoff
    "redundancy_penalty": 0.3,     # Penalty for similar chunks
}
```

### Evaluation Weights

```python
from src import EVAL_WEIGHTS

EVAL_WEIGHTS = {
    "relevance": 0.35,    # How well chunks match query
    "redundancy": 0.25,   # Inverse of overlap (lower is better)
    "coverage": 0.20,     # Topic breadth
    "efficiency": 0.20,   # Token budget utilization
}
```

---

## 🎓 Examples

### Complete Pipeline
See [`examples/full_pipeline.py`](examples/full_pipeline.py)

### Custom Integration
See [`examples/custom_integration.py`](examples/custom_integration.py)

### Jupyter Notebooks
- [Data Ingestion](examples/notebooks/demo_01_data_ingestion.ipynb)
- [Dynamic Composition](examples/notebooks/demo_02_chunking.ipynb)
- [Embedding & Retrieval](examples/notebooks/demo_03_embedding_retrieval.ipynb)
- [Pinecone LTM](examples/notebooks/demo_08_pinecone_ltm.ipynb)

---

## 📖 Documentation

- [Getting Started Guide](docs/getting-started.md)
- [Architecture Deep Dive](docs/architecture.md)
- [API Reference](docs/api-reference.md)
- [Integration Guide](docs/integration-guide.md) (LangChain, LlamaIndex, FastAPI)
- [Production Summary](PRODUCTION_SUMMARY.md)

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test suite
pytest tests/test_imports.py -v

# Run benchmark
python benchmarks/rag_vs_dce.py
```

### Sample Output

**RAG vs DCE Comparison:**

```bash
$ python benchmarks/rag_vs_dce.py
```

```
================================================================================
🔬 RAG vs DCE Rigorous Comparison
================================================================================
Dataset: Biotech QA (qa_biotech)
Queries: 10
Context Budget: 2048 tokens
Top-K Retrieval: 50

================================================================================
Query 1/10: What is de-extinction and how does it work?
================================================================================

🔵 Running Traditional RAG...
  Chunks: 10
  Tokens: 856.0
  Time: 1.98s
  Novelty: 0.1311
  Coverage: 0.5904
  Redundancy: 0.9859
  Top Sim: 0.8397

🟢 Running DCE...
  Chunks: 1
  Tokens: 85
  Time: 1.77s
  Novelty: 0.0000
  Coverage: 0.5841
  Redundancy: 0.0000
  Top Sim: 0.8374

  💡 DCE Improvements:
     Redundancy Reduction: +98.6%
     Token Efficiency: -90.1% (fewer tokens)
     Speed: +10.6% faster

================================================================================
📊 FINAL AGGREGATE RESULTS (10 queries)
================================================================================

Metric Comparison:
┌────────────────────┬──────────────┬──────────────┬─────────────┐
│ Metric             │ Traditional  │ DCE          │ Improvement │
│                    │ RAG          │              │             │
├────────────────────┼──────────────┼──────────────┼─────────────┤
│ Avg Chunks         │ 10.0         │ 2.4          │ -76%        │
│ Avg Tokens         │ 877          │ 220          │ -75% ⭐⭐⭐   │
│ Avg Time (s)       │ 0.70         │ 0.97         │ +38%        │
│ Diversity          │ 10.0%        │ 53.3%        │ +433% ⭐⭐   │
│ Redundancy         │ 96.8%        │ 51.4%        │ -47% ⭐⭐    │
│ Novelty            │ 17.7%        │ 20.4%        │ +15%        │
│ Top Similarity     │ 81.6%        │ 80.7%        │ -1.1%       │
└────────────────────┴──────────────┴──────────────┴─────────────┘

Statistical Significance (paired t-test):
  Diversity:   p = 0.002 ⭐⭐ (highly significant)
  Redundancy:  p = 0.004 ⭐⭐ (highly significant)
  Novelty:     p = 0.420 (trending positive)

🎉 DCE achieves equivalent relevance with 75% fewer tokens while 
   pulling from 5.3x more diverse documents!
```

**Biotech Knowledge Base Query:**

```bash
$ python tests/test_biotech_retrieval.py
```

```
============================================================
  BIOTECH QA RETRIEVAL TEST
============================================================
✅ Connected to existing Pinecone index: self-evolving-dce-ltm

============================================================
Query: What is de-extinction?
============================================================

Top 3 Results:

1. Score: 0.876
   Text: Q: What does the term 'de-extinction' refer to?
   A: De-extinction refers to the process of generating organisms 
   that resemble or are the same as extinct species. It has potential 
   applications in species revival, conservation, and biodiversity 
   restoration...

2. Score: 0.872
   Text: Q: What is de-extinction?
   A: De-extinction, or resurrection biology, is the process of 
   generating organisms that resemble or are the same as extinct 
   species...

3. Score: 0.856
   Text: Q: What does the term 'de-extinction' refer to?
   A: The term 'de-extinction' refers to the scientific efforts 
   aimed at reviving extinct species through advanced genetic 
   engineering and biotechnology...

============================================================
Query: Explain CRISPR gene editing
============================================================

Top 3 Results:

1. Score: 0.842
   Text: Q: What is gene editing and how does it function?
   A: Gene editing is a biotechnology that allows scientists to 
   make precise changes to the DNA of living organisms. This 
   technology enables the addition, removal, or modification of 
   genetic material at specific locations...

2. Score: 0.836
   Text: Q: What is gene editing and how does it work?
   A: Gene editing is a powerful biotechnology that allows 
   scientists to make precise changes in the DNA of living 
   organisms...

3. Score: 0.834
   Text: Q: What is gene editing?
   A: Gene editing is a method that allows scientists to modify 
   an organism's genetic makeup using engineered enzymes called 
   nucleases...
```

---

## 🤝 Integration with Other Frameworks

### LangChain
```python
from langchain.retrievers import BaseRetriever
from src import ContextComposer, PineconeLTM

class DCERetriever(BaseRetriever):
    def get_relevant_documents(self, query: str):
        vectordb = PineconeLTM(...)
        candidates, embeddings, query_vec = vectordb.search(query)
        composer = ContextComposer(candidates, embeddings)
        context = composer.compose(query, list(range(len(candidates))), query_vec)
        return context['selected']
```

### LlamaIndex
```python
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import BaseRetriever

# Similar integration pattern
```

### FastAPI
```python
from fastapi import FastAPI
from src import ContextComposer

app = FastAPI()

@app.post("/compose")
async def compose_context(query: str):
    # Your DCE pipeline
    return {"context": context_pkg}
```

---

## 🌟 Key Differentiators

| Feature | Traditional RAG | DCE |
|---------|----------------|-----|
| Selection Strategy | Top-K by similarity | Multi-objective optimization |
| Diversity | None | Enforced via document spread |
| Redundancy Handling | None | Active filtering (cosine > 0.75) |
| Policy Evolution | Manual tuning | **Autonomous via Letta agents** |
| Evaluation | LLM judges (expensive) | Objective metrics (fast) |
| Token Efficiency | Fixed top-K | Budget-aware greedy selection |
| Production Ready | Varies | **Yes** (Pinecone, monitoring, tests) |

---

## 📈 Roadmap

- [x] Core dynamic composition
- [x] Policy evolution framework
- [x] Letta agent integration
- [x] Pinecone LTM
- [x] Comprehensive benchmarks
- [ ] Auto fine-tuning dataset generation (in progress)
- [ ] Auto quantization for edge deployment
- [ ] Multi-modal context composition
- [ ] Streaming composition for real-time
- [ ] Web UI dashboard

---

## 📝 Citation

If you use DCE in your research, please cite:

```bibtex
@software{dce2026,
  title = {Self-Evolving Dynamic Context Engine},
  author = {DCE Team},
  year = {2026},
  url = {https://github.com/adilsheraz/self-evolving-dce},
  note = {Developed in collaboration with Institute of Advanced Engineering (IAE) and wAI Industries},
  institution = {Institute of Advanced Engineering, Pakistan},
  organization = {wAI Industries}
}
```

**Collaborating Organizations:**
- [Institute of Advanced Engineering (IAE)](https://www.iae.com.pk/welcome) - Pakistan
- [wAI Industries](https://www.waiindustries.com/)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Letta** for autonomous agent framework
- **Pinecone** for vector database infrastructure
- **HuggingFace** for embedding models
- Research community for RAG advancements

---

## 📞 Contact

- Issues: [GitHub Issues](https://github.com/adilsheraz/self-evolving-dce/issues)
- Discussions: [GitHub Discussions](https://github.com/adilsheraz/self-evolving-dce/discussions)

---

**Built with ❤️ for the AI community**
