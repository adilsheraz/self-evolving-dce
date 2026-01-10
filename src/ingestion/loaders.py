"""
MDN Content Loader - Extract and clean MDN documentation
Supports sparse git cloning and frontmatter parsing
"""
import json
import re
import hashlib
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any


class MDNLoader:
    """Load and parse MDN documentation from git repository"""
    
    FRONTMATTER_MAX_LINES = 12
    MDN_MACRO_RE = re.compile(r"\{\{[^{}]{1,200}\}\}")
    HTML_TAG_RE = re.compile(r"<[^>]+>")
    
    def __init__(
        self,
        repo_url: str = "https://github.com/mdn/content.git",
        repo_dir: Path = None,
        include_subdirs: List[str] = None
    ):
        """
        Initialize MDN loader
        
        Args:
            repo_url: Git repository URL for MDN content
            repo_dir: Local directory to clone repo (default: notebooks/mdn_content_repo)
            include_subdirs: Specific subdirectories to include (default: core, getting_started, howto)
        """
        from src.config import Config
        
        self.repo_url = repo_url
        self.repo_dir = repo_dir or (Config.NOTEBOOKS_DIR / "mdn_content_repo")
        
        self.base_dir = self.repo_dir / "files" / "en-us" / "learn_web_development"
        
        # Default subdirs to include
        if include_subdirs is None:
            self.include_subdirs = [
                self.base_dir / "core",
                self.base_dir / "getting_started",
                self.base_dir / "howto",
            ]
        else:
            self.include_subdirs = [self.base_dir / sd for sd in include_subdirs]
    
    def ensure_repo(self) -> None:
        """Clone MDN repository using sparse checkout (only needed directories)"""
        if self.repo_dir.exists() and (self.repo_dir / ".git").exists():
            print(f"✅ Repo already exists: {self.repo_dir}")
            return
        
        if self.repo_dir.exists():
            print(f"⚠️ Folder exists but not a git repo. Removing: {self.repo_dir}")
            shutil.rmtree(self.repo_dir)
        
        print(f"📥 Sparse cloning (learn_web_development only): {self.repo_url} → {self.repo_dir}")
        
        # Initialize repo with sparse checkout
        self.repo_dir.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(["git", "init"], cwd=str(self.repo_dir))
        subprocess.check_call(["git", "remote", "add", "origin", self.repo_url], cwd=str(self.repo_dir))
        subprocess.check_call(["git", "config", "core.sparseCheckout", "true"], cwd=str(self.repo_dir))
        
        # Specify only the directories we need
        sparse_file = self.repo_dir / ".git" / "info" / "sparse-checkout"
        sparse_file.parent.mkdir(parents=True, exist_ok=True)
        sparse_file.write_text(
            "files/en-us/learn_web_development/core/\n"
            "files/en-us/learn_web_development/getting_started/\n"
            "files/en-us/learn_web_development/howto/\n",
            encoding="utf-8"
        )
        
        # Pull only specified directories
        subprocess.check_call(["git", "pull", "--depth", "1", "origin", "main"], cwd=str(self.repo_dir))
        print("✅ Sparse clone complete (only learn_web_development dirs)")
    
    @staticmethod
    def parse_mdn_frontmatter(md_text: str) -> Tuple[Dict[str, str], str]:
        """
        Parse MDN's tab-separated frontmatter format
        
        Returns:
            Tuple of (metadata dict, body text)
        """
        lines = md_text.splitlines()
        meta = {}
        
        # Try MDN tab-separated "header row + value row"
        if len(lines) >= 2 and ("\t" in lines[0]) and ("\t" in lines[1]):
            headers = [h.strip() for h in lines[0].split("\t") if h.strip()]
            values = [v.strip() for v in lines[1].split("\t")]
            
            if len(headers) >= 2 and len(values) >= 2:
                for i, h in enumerate(headers):
                    if i < len(values):
                        meta[h] = values[i]
                
                # Body starts after the first blank line
                body_start = 2
                while body_start < len(lines) and lines[body_start].strip() != "":
                    body_start += 1
                while body_start < len(lines) and lines[body_start].strip() == "":
                    body_start += 1
                body = "\n".join(lines[body_start:])
                return meta, body
        
        # Fallback: no recognizable frontmatter
        return meta, md_text
    
    @classmethod
    def clean_mdn_text(cls, text: str) -> str:
        """Remove MDN macros, HTML tags, and normalize whitespace"""
        # Remove MDN macros like {{PreviousMenu(...)}}
        text = cls.MDN_MACRO_RE.sub("", text)
        
        # Remove HTML tags
        text = cls.HTML_TAG_RE.sub(" ", text)
        
        # Normalize whitespace
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        
        return text.strip()
    
    @staticmethod
    def extract_title(meta: Dict[str, str], body: str, path: Path) -> str:
        """Extract title from metadata, markdown H1, or filename"""
        if "title" in meta and meta["title"].strip():
            return meta["title"].strip()
        
        # Fallback: first markdown H1
        for line in body.splitlines()[:40]:
            if line.startswith("# "):
                return line[2:].strip()
        
        # Fallback: filename
        return path.parent.name.replace("_", " ").strip() or path.name
    
    @staticmethod
    def stable_doc_id(rel_path: str) -> str:
        """Generate stable document ID from relative path"""
        return hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:16]
    
    @staticmethod
    def build_mdn_url(rel_path_under_files: str) -> str:
        """Convert repo path to MDN documentation URL"""
        rp = rel_path_under_files.replace("\\", "/")
        if rp.startswith("files/en-us/"):
            rp = rp[len("files/en-us/"):]
        
        # Strip trailing "/index.md" or ".md"
        rp = rp.replace("/index.md", "").replace(".md", "")
        
        return f"https://developer.mozilla.org/en-US/docs/{rp}"
    
    def iter_markdown_files(self) -> List[Path]:
        """Find all markdown files in included subdirectories"""
        md_files = []
        for root_dir in self.include_subdirs:
            if not root_dir.exists():
                raise FileNotFoundError(f"Missing directory: {root_dir}")
            md_files.extend(sorted(root_dir.rglob("*.md")))
        return md_files
    
    def load_raw(self, output_path: Path = None) -> List[Dict[str, Any]]:
        """
        Load raw documents with unparsed frontmatter
        
        Args:
            output_path: Optional path to save raw JSONL output
            
        Returns:
            List of raw document dictionaries
        """
        self.ensure_repo()
        
        md_files = self.iter_markdown_files()
        print(f"📄 Found {len(md_files)} markdown files in scope")
        
        raw_docs = []
        for path in md_files:
            txt = path.read_text(encoding="utf-8", errors="ignore")
            meta, body = self.parse_mdn_frontmatter(txt)
            
            rel_under_repo = str(path.relative_to(self.repo_dir)).replace("\\", "/")
            doc_id = self.stable_doc_id(rel_under_repo)
            
            title = self.extract_title(meta, body, path)
            url = self.build_mdn_url(rel_under_repo)
            
            raw_docs.append({
                "doc_id": doc_id,
                "repo_rel_path": rel_under_repo,
                "source_url_guess": url,
                "meta": meta,
                "title": title,
                "body_raw": body
            })
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                for doc in raw_docs:
                    f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            print(f"✅ Saved raw docs: {output_path} (count={len(raw_docs)})")
        
        return raw_docs
    
    def load_clean(self, output_path: Path = None) -> List[Dict[str, Any]]:
        """
        Load cleaned documents ready for embedding
        
        Args:
            output_path: Optional path to save cleaned JSONL output
            
        Returns:
            List of cleaned document dictionaries
        """
        raw_docs = self.load_raw()
        
        clean_docs = []
        for doc in raw_docs:
            body_clean = self.clean_mdn_text(doc["body_raw"])
            text_for_kb = f"{doc['title']}\n\n{body_clean}".strip()
            
            clean_docs.append({
                "doc_id": doc["doc_id"],
                "repo_rel_path": doc["repo_rel_path"],
                "source_url_guess": doc["source_url_guess"],
                "title": doc["title"],
                "text": text_for_kb,
                "meta": doc["meta"],
            })
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                for doc in clean_docs:
                    f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            print(f"✅ Saved clean docs: {output_path} (count={len(clean_docs)})")
        
        return clean_docs


class DataLoader:
    """Generic data loader supporting multiple formats"""
    
    @staticmethod
    def load_jsonl(path: Path) -> List[Dict[str, Any]]:
        """Load documents from JSONL file"""
        docs = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(json.loads(line))
        return docs
    
    @staticmethod
    def save_jsonl(docs: List[Dict[str, Any]], path: Path) -> None:
        """Save documents to JSONL file"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for doc in docs:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    
    @staticmethod
    def load_json(path: Path) -> Any:
        """Load data from JSON file"""
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    
    @staticmethod
    def save_json(data: Any, path: Path) -> None:
        """Save data to JSON file"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
