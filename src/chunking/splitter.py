"""
Document Chunking - Split documents into retrieval-ready chunks
Uses heading-aware splitting with token limits
"""
import re
from typing import List, Dict, Any


def estimate_tokens(text: str) -> int:
    """
    Rough token estimation (1 token ≈ 0.75 words)
    
    Args:
        text: Input text
        
    Returns:
        Estimated token count
    """
    return max(1, int(len(text.split()) / 0.75))


class DocumentChunker:
    """
    Heading-aware document chunker for semantic splitting
    
    Strategy:
    - Primary boundary: Markdown headings (H1-H6)
    - Secondary boundary: Token limit (hard split if exceeded)
    - No overlap (can be added later as evolution)
    """
    
    HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)", re.MULTILINE)
    
    def __init__(
        self,
        max_chunk_tokens: int = 300,
        min_chunk_tokens: int = 80
    ):
        """
        Initialize chunker with token limits
        
        Args:
            max_chunk_tokens: Maximum tokens per chunk (hard limit)
            min_chunk_tokens: Minimum tokens to keep a chunk
        """
        self.max_chunk_tokens = max_chunk_tokens
        self.min_chunk_tokens = min_chunk_tokens
    
    def split_by_headings(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text by markdown headings
        
        Args:
            text: Document text with markdown headings
            
        Returns:
            List of sections with heading metadata
        """
        matches = list(self.HEADING_RE.finditer(text))
        
        if not matches:
            return [{"heading": None, "level": None, "text": text.strip()}]
        
        sections = []
        for i, match in enumerate(matches):
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            
            sections.append({
                "heading": match.group(2).strip(),
                "level": len(match.group(1)),
                "text": text[start:end].strip()
            })
        
        return sections
    
    def chunk_section(
        self,
        section: Dict[str, Any],
        doc_metadata: Dict[str, Any],
        chunk_id_start: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Chunk a single section, respecting token limits
        
        Args:
            section: Section dict with heading and text
            doc_metadata: Parent document metadata
            chunk_id_start: Starting chunk ID counter
            
        Returns:
            List of chunk dictionaries
        """
        if not section["text"]:
            return []
        
        tokens = estimate_tokens(section["text"])
        
        # Skip chunks that are too small
        if tokens < self.min_chunk_tokens:
            return []
        
        chunks = []
        chunk_counter = chunk_id_start
        
        # Section fits within limit - single chunk
        if tokens <= self.max_chunk_tokens:
            chunks.append({
                "chunk_id": f"chunk_{chunk_counter}",
                "doc_id": doc_metadata.get("doc_id", "unknown"),
                "title": doc_metadata.get("title", ""),
                "heading": section["heading"],
                "heading_level": section["level"],
                "text": section["text"],
                "tokens": tokens,
                "source_path": doc_metadata.get("repo_rel_path", "")
            })
            return chunks
        
        # Section exceeds limit - hard split by words
        words = section["text"].split()
        for i in range(0, len(words), self.max_chunk_tokens):
            sub_text = " ".join(words[i:i + self.max_chunk_tokens])
            sub_tokens = estimate_tokens(sub_text)
            
            if sub_tokens >= self.min_chunk_tokens:
                chunks.append({
                    "chunk_id": f"chunk_{chunk_counter}",
                    "doc_id": doc_metadata.get("doc_id", "unknown"),
                    "title": doc_metadata.get("title", ""),
                    "heading": section["heading"],
                    "heading_level": section["level"],
                    "text": sub_text,
                    "tokens": sub_tokens,
                    "source_path": doc_metadata.get("repo_rel_path", ""),
                    "is_split": True,  # Mark as split from larger section
                    "split_index": i // self.max_chunk_tokens
                })
                chunk_counter += 1
        
        return chunks
    
    def chunk_document(self, doc: Dict[str, Any], chunk_id_start: int = 0) -> List[Dict[str, Any]]:
        """
        Chunk a single document into multiple chunks
        
        Args:
            doc: Document dictionary with 'text' field
            chunk_id_start: Starting chunk ID counter
            
        Returns:
            List of chunk dictionaries
        """
        sections = self.split_by_headings(doc.get("text", ""))
        
        chunks = []
        chunk_counter = chunk_id_start
        
        for section in sections:
            section_chunks = self.chunk_section(section, doc, chunk_counter)
            chunks.extend(section_chunks)
            chunk_counter += len(section_chunks)
        
        return chunks
    
    def chunk_documents(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk multiple documents
        
        Args:
            docs: List of document dictionaries
            
        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        chunk_counter = 0
        
        for doc in docs:
            doc_chunks = self.chunk_document(doc, chunk_counter)
            all_chunks.extend(doc_chunks)
            chunk_counter += len(doc_chunks)
        
        return all_chunks
    
    def get_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about chunks
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Statistics dictionary
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "avg_tokens": 0,
                "min_tokens": 0,
                "max_tokens": 0,
                "chunks_per_doc": 0
            }
        
        token_counts = [c["tokens"] for c in chunks]
        doc_ids = [c["doc_id"] for c in chunks]
        unique_docs = len(set(doc_ids))
        
        return {
            "total_chunks": len(chunks),
            "avg_tokens": sum(token_counts) / len(token_counts),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "total_tokens": sum(token_counts),
            "unique_documents": unique_docs,
            "chunks_per_doc": len(chunks) / unique_docs if unique_docs > 0 else 0
        }
