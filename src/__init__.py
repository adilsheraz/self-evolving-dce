"""
Self-Evolving Dynamic Context Engineering (DCE) - Core Library

A research framework for dynamic context composition and agent-guided policy experiments.

Key Features:
- Dynamic context composition with multi-objective optimization
- Autonomous policy evolution via Letta agents
- Long-term memory with Pinecone vector database
- Automatic fine-tuning dataset generation
- Comprehensive evaluation metrics

Usage:
    from src import ContextComposer, ContextEvaluator, SelfEvolvingLoop
"""

__version__ = "0.1.0"
__author__ = "DCE Team"

# Core modules
from .ingestion import DataLoader, MDNLoader
from .chunking import DocumentChunker, estimate_tokens
from .embedding import EmbeddingProvider, HuggingFaceEmbedder
from .context import ContextComposer, render_context_text
from .evaluation import ContextEvaluator, EVAL_WEIGHTS
from .evolution import PolicyEvolver, build_policy_variant
from .autoft import QAGenerator, QAValidator
from .vectordb import PineconeLTM
from .letta import create_policy_strategist, LettaPolicyStrategist

# Orchestration
from .orchestration import SelfEvolvingLoop

# Configuration
from .config import Config

__all__ = [
    # Core engine
    "Config",
    
    # Ingestion
    "DataLoader",
    "MDNLoader",
    
    # Chunking
    "DocumentChunker",
    "estimate_tokens",
    
    # Embedding
    "EmbeddingProvider",
    "HuggingFaceEmbedder",
    
    # Context composition
    "ContextComposer",
    "render_context_text",
    
    # Evaluation
    "ContextEvaluator",
    "EVAL_WEIGHTS",
    
    # Evolution
    "PolicyEvolver",
    "build_policy_variant",
    
    # AutoFT
    "QAGenerator",
    "QAValidator",
    
    # Vector database
    "PineconeLTM",
    
    # Letta agents
    "create_policy_strategist",
    "LettaPolicyStrategist",
    
    # Orchestration
    "SelfEvolvingLoop",
]

