"""
AutoFT Q/A Dataset Generator

Generates supervised fine-tuning datasets from documentation chunks using LLMs.

Key Features:
- Grounded generation (answers derived from chunk content)
- Multi-layer quality validation
- Automated metrics (length, format, grounding)
- Configurable difficulty levels
- Export ready for downstream AutoFT
"""

import json
import time
import requests
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional


class QAGenerator:
    """
    Generate Q/A pairs from documentation chunks using LLM API.
    
    Supports:
    - Groq API (llama-3.3-70b-versatile)
    - Configurable retry logic
    - Rate limiting
    """
    
    SYSTEM_PROMPT = """You are an expert educational content creator.

Your task is to generate high-quality question-answer pairs from documentation chunks.

Rules:
1. Generate 2-4 diverse questions per chunk
2. Questions should test understanding at different levels (recall, comprehension, application)
3. Answers must be directly answerable from the chunk content
4. Do NOT add external knowledge
5. Keep questions clear and concise
6. Keep answers accurate and complete

Output format (JSON array):
[
  {
    "question": "...",
    "answer": "...",
    "difficulty": "easy|medium|hard"
  }
]
"""
    
    USER_PROMPT_TEMPLATE = """Generate question-answer pairs from this documentation chunk:

---
{chunk_text}
---

Generate 2-4 questions with answers. Return JSON array only."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        api_url: str = "https://api.groq.com/openai/v1/chat/completions",
        rate_limit_delay: float = 5.0
    ):
        """
        Initialize Q/A generator.
        
        Args:
            api_key: Groq API key
            model: Model identifier
            api_url: API endpoint URL
            rate_limit_delay: Delay between requests (seconds)
        """
        self.api_key = api_key
        self.model = model
        self.api_url = api_url
        self.rate_limit_delay = rate_limit_delay
    
    def generate_qa_pairs(self, chunk_text: str, retry: int = 3) -> List[Dict[str, Any]]:
        """
        Generate Q/A pairs from chunk text.
        
        Args:
            chunk_text: Documentation chunk content
            retry: Number of retry attempts
            
        Returns:
            List of Q/A pair dictionaries
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": self.USER_PROMPT_TEMPLATE.format(chunk_text=chunk_text)}
            ],
            "temperature": 0.7,
            "max_tokens": 800,
            "response_format": {"type": "json_object"}
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        for attempt in range(retry):
            try:
                response = requests.post(self.api_url, json=payload, headers=headers, timeout=30)
                response.raise_for_status()
                
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                qa_pairs = json.loads(content)
                
                # Handle both array and object with array
                if isinstance(qa_pairs, dict) and "questions" in qa_pairs:
                    qa_pairs = qa_pairs["questions"]
                
                return qa_pairs if isinstance(qa_pairs, list) else []
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    # Rate limit - exponential backoff
                    wait_time = 3 * (2 ** attempt)
                    time.sleep(wait_time)
                elif e.response.status_code == 400:
                    # Bad request - likely JSON validation error
                    return []
                else:
                    if attempt < retry - 1:
                        time.sleep(2)
                        
            except (json.JSONDecodeError, Exception):
                if attempt < retry - 1:
                    time.sleep(2)
        
        return []
    
    def generate_dataset(
        self,
        chunks: List[Dict[str, Any]],
        target_samples: int = 500,
        verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate Q/A dataset from chunk list.
        
        Args:
            chunks: List of chunk dictionaries (must have 'text' field)
            target_samples: Target number of Q/A pairs
            verbose: Print progress
            
        Returns:
            List of dataset records
        """
        dataset_records = []
        failed_chunks = []
        
        start_time = time.time()
        
        for i, chunk in enumerate(chunks):
            if verbose:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(chunks) - i) / rate if rate > 0 else 0
                print(f"Processing {i+1}/{len(chunks)} | Q/A pairs: {len(dataset_records)} | "
                      f"Failed: {len(failed_chunks)} | ETA: {eta/60:.1f}min", end="\r")
            
            qa_pairs = self.generate_qa_pairs(chunk.get("text", ""))
            
            if qa_pairs:
                for qa in qa_pairs:
                    dataset_records.append({
                        "question": qa.get("question", ""),
                        "answer": qa.get("answer", ""),
                        "difficulty": qa.get("difficulty", "medium"),
                        "chunk_id": chunk.get("chunk_id", ""),
                        "chunk_text": chunk.get("text", ""),
                        "source_url": chunk.get("url", chunk.get("source_path", "")),
                        "source_title": chunk.get("title", "")
                    })
            else:
                failed_chunks.append(chunk.get("chunk_id", i))
            
            time.sleep(self.rate_limit_delay)
            
            if len(dataset_records) >= target_samples:
                if verbose:
                    print(f"\n✅ Reached target of {target_samples} Q/A pairs")
                break
        
        if verbose:
            print(f"\n✅ Generated {len(dataset_records)} Q/A pairs from {len(chunks)} chunks")
            print(f"⚠️ Failed chunks: {len(failed_chunks)}")
            print(f"⏱️ Total time: {(time.time() - start_time)/60:.1f} minutes")
        
        return dataset_records


class QAValidator:
    """
    Validate Q/A pair quality using automated metrics.
    
    Validation layers:
    1. Length checks (min/max)
    2. Format validation (question marks, etc.)
    3. Content quality (no refusals, meta-questions)
    4. Grounding check (overlap with source chunk)
    """
    
    DEFAULT_CONFIG = {
        "min_question_length": 10,
        "min_answer_length": 20,
        "max_answer_length": 1500,
        "min_grounding_overlap": 0.1,
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize validator.
        
        Args:
            config: Validation configuration (uses defaults if not provided)
        """
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
    
    def validate_qa_pair(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate single Q/A pair.
        
        Args:
            record: Q/A record dictionary
            
        Returns:
            Validation result with 'valid' flag and 'issues' list
        """
        question = record.get("question", "")
        answer = record.get("answer", "")
        
        metrics = {
            "valid": True,
            "issues": []
        }
        
        # Length checks
        if len(question) < self.config["min_question_length"]:
            metrics["issues"].append("question_too_short")
            metrics["valid"] = False
        
        if len(answer) < self.config["min_answer_length"]:
            metrics["issues"].append("answer_too_short")
            metrics["valid"] = False
        
        if len(answer) > self.config["max_answer_length"]:
            metrics["issues"].append("answer_too_long")
        
        # Question quality
        if not question.strip().endswith("?"):
            metrics["issues"].append("missing_question_mark")
        
        if question.lower().startswith(("generate", "create", "write")):
            metrics["issues"].append("meta_question")
            metrics["valid"] = False
        
        # Answer quality
        if answer.lower().startswith(("i don't", "i cannot", "sorry")):
            metrics["issues"].append("refusal_answer")
            metrics["valid"] = False
        
        # Hallucination indicators
        hallucination_phrases = [
            "as an ai", "i'm not sure", "in general",
            "it depends", "typically", "usually"
        ]
        if any(phrase in answer.lower() for phrase in hallucination_phrases):
            metrics["issues"].append("possible_hallucination")
        
        # Grounding check
        chunk_text = record.get("chunk_text", "")
        if chunk_text:
            chunk_words = set(chunk_text.lower().split())
            answer_words = set(answer.lower().split())
            overlap = len(chunk_words & answer_words) / max(len(answer_words), 1)
            
            if overlap < self.config["min_grounding_overlap"]:
                metrics["issues"].append("low_grounding")
                metrics["valid"] = False
        
        return metrics
    
    def validate_dataset(
        self,
        records: List[Dict[str, Any]],
        verbose: bool = True
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Validate entire dataset.
        
        Args:
            records: List of Q/A records
            verbose: Print statistics
            
        Returns:
            Tuple of (validated_records, quality_stats)
        """
        validated_records = []
        quality_stats = {
            "total": len(records),
            "valid": 0,
            "invalid": 0,
            "issues_count": {}
        }
        
        for record in records:
            validation = self.validate_qa_pair(record)
            record["validation"] = validation
            validated_records.append(record)
            
            if validation["valid"]:
                quality_stats["valid"] += 1
            else:
                quality_stats["invalid"] += 1
            
            for issue in validation["issues"]:
                quality_stats["issues_count"][issue] = quality_stats["issues_count"].get(issue, 0) + 1
        
        if verbose:
            print(f"✅ Automated validation complete:")
            print(f"   Valid: {quality_stats['valid']} ({quality_stats['valid']/quality_stats['total']*100:.1f}%)")
            print(f"   Invalid: {quality_stats['invalid']}")
            if quality_stats["issues_count"]:
                print(f"\n⚠️ Common issues:")
                for issue, count in sorted(quality_stats["issues_count"].items(), key=lambda x: -x[1])[:5]:
                    print(f"   {issue}: {count}")
        
        return validated_records, quality_stats
    
    @staticmethod
    def export_dataset(
        records: List[Dict[str, Any]],
        output_path: Path,
        include_validation: bool = False
    ):
        """
        Export dataset to JSONL file.
        
        Args:
            records: List of Q/A records
            output_path: Output file path
            include_validation: Include validation metadata in export
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with output_path.open("w", encoding="utf-8") as f:
            for record in records:
                clean_record = {
                    "question": record["question"],
                    "answer": record["answer"],
                    "difficulty": record.get("difficulty", "medium"),
                    "chunk_id": record.get("chunk_id", ""),
                    "source_url": record.get("source_url", ""),
                    "source_title": record.get("source_title", "")
                }
                
                if include_validation and "validation" in record:
                    clean_record["validation"] = record["validation"]
                
                f.write(json.dumps(clean_record, ensure_ascii=False) + "\n")
        
        print(f"✅ Exported {len(records)} records to {output_path}")
