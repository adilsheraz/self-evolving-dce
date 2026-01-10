"""
AutoFT Dataset Generation Module

Generates high-quality Q/A datasets from documentation chunks using LLMs.
"""

from .generator import QAGenerator, QAValidator

__all__ = ["QAGenerator", "QAValidator"]
