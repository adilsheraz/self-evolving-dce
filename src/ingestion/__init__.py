"""Data Ingestion Module - Load and parse documents for DCE pipeline"""

from .loaders import DataLoader, MDNLoader

__all__ = ["DataLoader", "MDNLoader"]
