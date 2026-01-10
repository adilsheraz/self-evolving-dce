"""
New Dataset Integration Guide: American Genetic De-Extinction Data

This script shows how to use the auto_rag_pipeline.py with the new dataset.
"""
import json
from pathlib import Path

# ==============================================================================
# DATASET ANALYSIS
# ==============================================================================
# Your new dataset: americangenetic_page_summaries.jsonl
# Location: F:\PMYP-Training\Data by Ammar\colossal\pipeline_output_final\embeddings\
#
# Structure:
# {
#   "document": "americangenetic",
#   "page": 1,
#   "executive_summary": "Long text...",
#   "key_entities": ["entity1", "entity2", ...],
#   "generated_at": "2025-12-27T22:44:10.069574"
# }

# ==============================================================================
# OPTION 1: Direct Use (If Format is Compatible)
# ==============================================================================
# If your JSONL has 'text' or 'content' field, it works directly!

# Check if format is compatible
def check_dataset_compatibility(file_path: str):
    """Check if dataset can be used directly with auto_rag_pipeline"""
    with open(file_path, 'r', encoding='utf-8') as f:
        first_line = json.loads(f.readline())
    
    print("Dataset fields:", list(first_line.keys()))
    
    # Required fields: 'text' or 'content'
    # Optional fields: 'doc_id', 'title', 'metadata'
    
    has_text = 'text' in first_line or 'content' in first_line
    if not has_text:
        print("⚠️ Dataset needs transformation (no 'text' or 'content' field)")
        print("   Available text field: 'executive_summary'")
        return False
    else:
        print("✅ Dataset compatible - can use directly!")
        return True

# ==============================================================================
# OPTION 2: Transform Dataset (Your Case)
# ==============================================================================
# Your dataset has 'executive_summary' instead of 'text'
# Need simple transformation

def transform_americangenetic_dataset(
    input_path: str,
    output_path: str = "data/americangenetic_docs.jsonl"
):
    """
    Transform americangenetic dataset to DCE-compatible format
    
    Args:
        input_path: Path to americangenetic_page_summaries.jsonl
        output_path: Path for transformed output
    """
    import json
    from pathlib import Path
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    transformed_docs = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            record = json.loads(line)
            
            # Transform to DCE format
            transformed = {
                # Required fields
                "doc_id": f"americangenetic_page_{record['page']:03d}",
                "title": f"De-Extinction Technology - Page {record['page']}",
                "text": record['executive_summary'],  # Main content
                
                # Optional metadata
                "metadata": {
                    "document": record['document'],
                    "page": record['page'],
                    "key_entities": record['key_entities'],
                    "generated_at": record['generated_at'],
                    "source": "American Genetic De-Extinction Research"
                }
            }
            
            transformed_docs.append(transformed)
    
    # Save transformed dataset
    with open(output_path, 'w', encoding='utf-8') as f:
        for doc in transformed_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    print(f"✅ Transformed {len(transformed_docs)} documents")
    print(f"   Output: {output_path}")
    return output_path

# ==============================================================================
# OPTION 3: Run Full Pipeline
# ==============================================================================

def run_pipeline_with_new_dataset():
    """Complete workflow for americangenetic dataset"""
    
    print("="*60)
    print("  INTEGRATING AMERICAN GENETIC DATASET INTO DCE")
    print("="*60)
    
    # Step 1: Transform dataset
    print("\n[Step 1] Transforming dataset...")
    input_file = r"F:\PMYP-Training\Data by Ammar\colossal\pipeline_output_final\embeddings\americangenetic_page_summaries.jsonl"
    output_file = "data/americangenetic_docs.jsonl"
    
    transform_americangenetic_dataset(input_file, output_file)
    
    # Step 2: Run auto-RAG pipeline
    print("\n[Step 2] Running auto-RAG pipeline...")
    print("\nExecute this command:")
    print(f"""
    python scripts/auto_rag_pipeline.py \\
        --dataset {output_file} \\
        --max-chunk-tokens 400 \\
        --index-name americangenetic-dce \\
        --test-query "What are the ethical considerations of de-extinction?"
    """)
    
    print("\n[Step 3] Pipeline will automatically:")
    print("  ✅ Load 17 documents")
    print("  ✅ Chunk into ~50-100 chunks (depends on content length)")
    print("  ✅ Generate embeddings (384-dim vectors)")
    print("  ✅ Upload to Pinecone index 'americangenetic-dce'")
    print("  ✅ Test retrieval with query")
    
    print("\n[Step 4] After pipeline completes:")
    print("  ✅ Query the knowledge base")
    print("  ✅ Run policy evolution experiments")
    print("  ✅ Generate AutoFT test datasets")

# ==============================================================================
# ENHANCED: Multi-Field Content Strategy
# ==============================================================================

def transform_with_entity_enrichment(
    input_path: str,
    output_path: str = "data/americangenetic_enriched.jsonl"
):
    """
    Enhanced transformation that includes key entities in the text
    This improves retrieval by making entity mentions searchable
    """
    import json
    from pathlib import Path
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    transformed_docs = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            record = json.loads(line)
            
            # Enhance text with entity context
            entity_text = "\n\nKey Topics: " + ", ".join(record['key_entities'][:20])
            enriched_text = record['executive_summary'] + entity_text
            
            transformed = {
                "doc_id": f"americangenetic_page_{record['page']:03d}",
                "title": f"De-Extinction Technology - Page {record['page']}",
                "text": enriched_text,  # Enhanced with entities
                "metadata": {
                    "document": record['document'],
                    "page": record['page'],
                    "num_entities": len(record['key_entities']),
                    "generated_at": record['generated_at'],
                    "source": "American Genetic De-Extinction Research",
                    "all_entities": record['key_entities']  # Keep for reference
                }
            }
            
            transformed_docs.append(transformed)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for doc in transformed_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    print(f"✅ Created enriched dataset: {len(transformed_docs)} documents")
    print(f"   Average entities per page: {sum(len(d['metadata']['all_entities']) for d in transformed_docs) / len(transformed_docs):.1f}")
    return output_path

# ==============================================================================
# USAGE EXAMPLES
# ==============================================================================

if __name__ == "__main__":
    # Example 1: Basic transformation
    print("\n=== EXAMPLE 1: Basic Transformation ===")
    transform_americangenetic_dataset(
        input_path=r"F:\PMYP-Training\Data by Ammar\colossal\pipeline_output_final\embeddings\americangenetic_page_summaries.jsonl",
        output_path="data/americangenetic_docs.jsonl"
    )
    
    # Example 2: Entity-enriched transformation
    print("\n=== EXAMPLE 2: Entity-Enriched Transformation ===")
    transform_with_entity_enrichment(
        input_path=r"F:\PMYP-Training\Data by Ammar\colossal\pipeline_output_final\embeddings\americangenetic_page_summaries.jsonl",
        output_path="data/americangenetic_enriched.jsonl"
    )
    
    # Example 3: Show command to run pipeline
    print("\n=== EXAMPLE 3: Run Auto-RAG Pipeline ===")
    run_pipeline_with_new_dataset()
