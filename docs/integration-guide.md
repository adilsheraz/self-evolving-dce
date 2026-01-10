# Integration Guide

Learn how to integrate DCE with popular frameworks and custom applications.

---

## Table of Contents

- [LangChain Integration](#langchain-integration)
- [LlamaIndex Integration](#llamaindex-integration)
- [FastAPI Integration](#fastapi-integration)
- [Custom Framework Integration](#custom-framework-integration)
- [Production Deployment](#production-deployment)

---

## LangChain Integration

DCE can be used as a custom retriever in LangChain.

### Step 1: Create Custom Retriever

```python
from langchain.retrievers import BaseRetriever
from langchain.schema import Document
from typing import List
from src import ContextComposer, PineconeLTM

class DCERetriever(BaseRetriever):
    """DCE-powered retriever for LangChain."""
    
    def __init__(self, api_key: str, index_name: str, policy: dict = None):
        self.vectordb = PineconeLTM(api_key=api_key, index_name=index_name)
        self.policy = policy or {
            "w_retrieval": 0.55,
            "w_novelty": 0.25,
            "w_doc_diversity": 0.20,
            "budget_tokens": 2048,
            "redundancy_threshold": 0.75
        }
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        # High-recall retrieval
        candidates, embeddings, query_vec = self.vectordb.search(query, top_k=50)
        
        # Dynamic composition
        composer = ContextComposer(
            chunks=candidates,
            chunk_embeddings=embeddings,
            policy=self.policy
        )
        
        context_pkg = composer.compose(
            query=query,
            candidate_idxs=list(range(len(candidates))),
            query_vec=query_vec
        )
        
        # Convert to LangChain Documents
        return [
            Document(
                page_content=chunk['text'],
                metadata={
                    "chunk_id": chunk['chunk_id'],
                    "doc_id": chunk['doc_id'],
                    "position": chunk['position']
                }
            )
            for chunk in context_pkg['selected']
        ]
    
    async def aget_relevant_documents(self, query: str) -> List[Document]:
        # Async version (if needed)
        return self.get_relevant_documents(query)
```

### Step 2: Use in LangChain Chain

```python
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Initialize retriever
retriever = DCERetriever(
    api_key="pcsk_...",
    index_name="dce-biotech"
)

# Create QA chain
llm = ChatOpenAI(model="gpt-4", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

# Ask question
response = qa_chain.run("What is CRISPR gene editing?")
print(response)
```

### Step 3: Advanced - Custom Chain

```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Custom prompt with DCE context
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a biotech expert. Use the following context to answer the question.

Context:
{context}

Question: {question}

Answer:"""
)

# Chain with DCE retriever
class DCEChain:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm_chain = LLMChain(llm=llm, prompt=prompt)
    
    def run(self, question: str):
        # Get DCE-composed context
        docs = self.retriever.get_relevant_documents(question)
        context = "\n\n".join([f"[{i+1}] {doc.page_content}" 
                               for i, doc in enumerate(docs)])
        
        # Generate response
        return self.llm_chain.run(context=context, question=question)

# Usage
dce_chain = DCEChain(retriever=retriever, llm=llm)
answer = dce_chain.run("Explain de-extinction technology")
```

---

## LlamaIndex Integration

DCE can replace the default retriever in LlamaIndex.

### Step 1: Create Custom Retriever

```python
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from typing import List
from src import ContextComposer, PineconeLTM

class DCERetriever(BaseRetriever):
    """DCE retriever for LlamaIndex."""
    
    def __init__(self, api_key: str, index_name: str, policy: dict = None):
        super().__init__()
        self.vectordb = PineconeLTM(api_key=api_key, index_name=index_name)
        self.policy = policy or {
            "w_retrieval": 0.55,
            "w_novelty": 0.25,
            "w_doc_diversity": 0.20,
            "budget_tokens": 2048
        }
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query = query_bundle.query_str
        
        # High-recall retrieval
        candidates, embeddings, query_vec = self.vectordb.search(query, top_k=50)
        
        # Dynamic composition
        composer = ContextComposer(
            chunks=candidates,
            chunk_embeddings=embeddings,
            policy=self.policy
        )
        
        context_pkg = composer.compose(
            query=query,
            candidate_idxs=list(range(len(candidates))),
            query_vec=query_vec
        )
        
        # Convert to LlamaIndex nodes
        from llama_index.core.schema import TextNode
        
        nodes_with_scores = []
        for chunk in context_pkg['selected']:
            node = TextNode(
                text=chunk['text'],
                id_=chunk['chunk_id'],
                metadata=chunk['metadata']
            )
            # Use retrieval score from context package
            score = chunk.get('retrieval_score', 1.0)
            nodes_with_scores.append(NodeWithScore(node=node, score=score))
        
        return nodes_with_scores
```

### Step 2: Use in Query Engine

```python
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai import OpenAI

# Initialize
retriever = DCERetriever(api_key="pcsk_...", index_name="dce-biotech")
llm = OpenAI(model="gpt-4", temperature=0)

# Create query engine
query_engine = RetrieverQueryEngine.from_args(
    retriever=retriever,
    llm=llm
)

# Query
response = query_engine.query("What is CRISPR?")
print(response.response)
```

---

## FastAPI Integration

Build a production API with DCE.

### Step 1: API Implementation

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from src import ContextComposer, PineconeLTM
import os

app = FastAPI(title="DCE API", version="1.0.0")

# Models
class QueryRequest(BaseModel):
    query: str
    top_k: int = 50
    budget_tokens: int = 2048
    namespace: str = "v1"

class ContextResponse(BaseModel):
    selected_chunks: List[Dict]
    used_tokens: int
    metrics: Dict
    query: str

# Initialize (once at startup)
vectordb = None

@app.on_event("startup")
async def startup():
    global vectordb
    vectordb = PineconeLTM(
        api_key=os.getenv("PINECONE_API_KEY"),
        index_name="dce-biotech"
    )

# Endpoint
@app.post("/compose", response_model=ContextResponse)
async def compose_context(request: QueryRequest):
    try:
        # Retrieve candidates
        candidates, embeddings, query_vec = vectordb.search(
            query=request.query,
            top_k=request.top_k,
            namespace=request.namespace
        )
        
        # Compose context
        composer = ContextComposer(
            chunks=candidates,
            chunk_embeddings=embeddings,
            policy={
                "w_retrieval": 0.55,
                "w_novelty": 0.25,
                "w_doc_diversity": 0.20,
                "budget_tokens": request.budget_tokens,
                "redundancy_threshold": 0.75
            }
        )
        
        context_pkg = composer.compose(
            query=request.query,
            candidate_idxs=list(range(len(candidates))),
            query_vec=query_vec
        )
        
        return ContextResponse(
            selected_chunks=context_pkg['selected'],
            used_tokens=context_pkg['used_tokens'],
            metrics=context_pkg['metrics'],
            query=request.query
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check
@app.get("/health")
async def health():
    return {"status": "ok"}
```

### Step 2: Run Server

```bash
# Install dependencies
pip install fastapi uvicorn

# Run
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Step 3: Client Usage

```python
import requests

response = requests.post(
    "http://localhost:8000/compose",
    json={
        "query": "What is CRISPR?",
        "budget_tokens": 2048
    }
)

data = response.json()
print(f"Selected {len(data['selected_chunks'])} chunks")
print(f"Metrics: {data['metrics']}")
```

---

## Custom Framework Integration

Integrate DCE into any custom application.

### Pattern 1: Direct Import

```python
from src import ContextComposer, PineconeLTM

class MyCustomRAG:
    def __init__(self):
        self.vectordb = PineconeLTM(...)
        self.policy = {...}
    
    def query(self, question: str):
        # Your custom logic
        candidates, embeddings, query_vec = self.vectordb.search(question)
        
        composer = ContextComposer(
            chunks=candidates,
            chunk_embeddings=embeddings,
            policy=self.policy
        )
        
        context_pkg = composer.compose(question, list(range(len(candidates))), query_vec)
        
        # Use context_pkg in your downstream pipeline
        return self.generate_response(context_pkg)
```

### Pattern 2: Standalone Module

```python
# Import only what you need
from src.context import ContextComposer
from src.evaluation import ContextEvaluator

# Bring your own retrieval
def my_retriever(query):
    # Your custom retrieval logic
    return candidates, embeddings, query_vec

# Compose with DCE
candidates, embeddings, query_vec = my_retriever("query")
composer = ContextComposer(candidates, embeddings)
context_pkg = composer.compose("query", list(range(len(candidates))), query_vec)

# Evaluate
evaluator = ContextEvaluator()
metrics = evaluator.evaluate(context_pkg)
```

### Pattern 3: Policy Evolution in Custom Loop

```python
from src.evolution import PolicyEvolver
from src.letta import create_policy_strategist

# Your existing system
class MyRAGSystem:
    def __init__(self):
        self.policy = {...}
    
    def evolve_policy(self, test_queries):
        # Use DCE evolution
        evolver = PolicyEvolver(
            baseline_policy=self.policy,
            baseline_contexts=self.get_baseline_contexts()
        )
        
        new_policy = evolver.evolve(queries=test_queries)
        self.policy = new_policy  # Update your system
```

---

## Production Deployment

### Docker Container

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ ./src/
COPY main.py .

# Expose port
EXPOSE 8000

# Run
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dce-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: dce-api
  template:
    metadata:
      labels:
        app: dce-api
    spec:
      containers:
      - name: dce-api
        image: your-registry/dce-api:v1.0.0
        ports:
        - containerPort: 8000
        env:
        - name: PINECONE_API_KEY
          valueFrom:
            secretKeyRef:
              name: dce-secrets
              key: pinecone-api-key
        - name: LETTA_API_KEY
          valueFrom:
            secretKeyRef:
              name: dce-secrets
              key: letta-api-key
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
---
apiVersion: v1
kind: Service
metadata:
  name: dce-api-service
spec:
  selector:
    app: dce-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Environment Variables

```bash
# Production .env
PINECONE_API_KEY=pcsk_...
PINECONE_INDEX_NAME=dce-production
LETTA_API_KEY=letta_sk_...
HF_TOKEN=hf_...

# Optional
LOG_LEVEL=INFO
SENTRY_DSN=https://...
```

---

## Best Practices

### 1. Caching

Cache embeddings to reduce API calls:

```python
import pickle

# Save
with open("embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)

# Load
with open("embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)
```

### 2. Batch Processing

Process large datasets in batches:

```python
def batch_upsert(chunks, embeddings, batch_size=100):
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        batch_embeddings = embeddings[i:i+batch_size]
        vectordb.upsert_chunks(batch_chunks, batch_embeddings)
```

### 3. Error Handling

Implement robust error handling:

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def safe_search(query):
    try:
        return vectordb.search(query)
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise
```

### 4. Monitoring

Add logging and metrics:

```python
import logging
import time

logger = logging.getLogger(__name__)

def monitored_compose(query, candidates, embeddings, query_vec):
    start = time.time()
    
    try:
        composer = ContextComposer(candidates, embeddings)
        context_pkg = composer.compose(query, list(range(len(candidates))), query_vec)
        
        duration = time.time() - start
        logger.info(f"Composed context in {duration:.2f}s")
        logger.info(f"Selected {len(context_pkg['selected'])} chunks")
        
        return context_pkg
    
    except Exception as e:
        logger.error(f"Composition failed: {e}")
        raise
```

---

## Integration Checklist

- [ ] Install DCE dependencies
- [ ] Set up environment variables
- [ ] Initialize Pinecone index
- [ ] Upload initial dataset
- [ ] Test basic query flow
- [ ] Integrate with your framework
- [ ] Add error handling
- [ ] Implement caching strategy
- [ ] Set up logging and monitoring
- [ ] Test in staging environment
- [ ] Deploy to production

---

**Next**: [Letta Agents Documentation →](letta-agents.md)
