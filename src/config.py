"""
Configuration module for Self-Evolving DCE
Loads environment variables and provides centralized config access
"""
from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()


class Config:
    """Central configuration class for all project settings"""
    
    # ===== Directory Paths =====
    REPO_ROOT = Path(__file__).parent.parent
    DATA_DIR = REPO_ROOT / "data"
    OUTPUTS_DIR = REPO_ROOT / "outputs"
    NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
    LOGS_DIR = REPO_ROOT / "logs"
    STATE_DIR = REPO_ROOT / "state"
    CONFIG_DIR = REPO_ROOT / "config"
    
    # MVP Knowledge Base paths (legacy)
    MVP_KB_OUTPUTS = NOTEBOOKS_DIR / "mvp_kb_outputs"
    
    # ===== API Keys (from environment) =====
    HF_TOKEN = os.getenv("HF_TOKEN")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    LETTA_API_KEY = os.getenv("LETTA_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    # ===== Vector Database Settings =====
    PINECONE_INDEX_NAME = "dce-mvp"
    PINECONE_DIMENSION = 384
    PINECONE_METRIC = "cosine"
    
    # ===== Embedding Settings =====
    DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION = 384
    
    # ===== Chunking Settings =====
    DEFAULT_CHUNK_SIZE = 512
    DEFAULT_CHUNK_OVERLAP = 64
    
    # ===== Retrieval Settings =====
    DEFAULT_TOP_K = 5
    
    @classmethod
    def validate(cls):
        """Validate that required environment variables are set"""
        missing = []
        if not cls.HF_TOKEN:
            missing.append("HF_TOKEN")
        if not cls.PINECONE_API_KEY:
            missing.append("PINECONE_API_KEY")
        
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}\n"
                f"Please copy .env.example to .env and fill in the values."
            )
    
    @classmethod
    def ensure_dirs(cls):
        """Create necessary directories if they don't exist"""
        for dir_path in [cls.DATA_DIR, cls.OUTPUTS_DIR, cls.LOGS_DIR, cls.STATE_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
