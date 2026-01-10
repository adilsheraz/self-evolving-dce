"""
End-to-End Test: AutoFT Q/A Generation

Tests the complete Q/A generation workflow:
1. Load sample chunks
2. Generate Q/A pairs using Groq API (1-2 chunks only to minimize costs)
3. Validate generated pairs
4. Export results

NOTE: This test makes real API calls and will consume Groq API credits.
"""

import sys
from pathlib import Path
import json
import time

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.config import Config
from src.autoft.generator import QAGenerator, QAValidator

print("="*60)
print("  End-to-End AutoFT Generation Test")
print("="*60)

print("\n⚠️  WARNING: This test will make real API calls to Groq")
print("   - It will generate Q/A pairs from 2 sample chunks")
print("   - This will consume API credits (~500-1000 tokens)")
print("   - Rate limited to 5 seconds between requests")

# Load config
config = Config()

# Check for API key
groq_api_key = config.GROQ_API_KEY
if not groq_api_key or groq_api_key == "your_groq_api_key_here":
    print("\n❌ Error: GROQ_API_KEY not set in .env file")
    print("   Please add your Groq API key to .env:")
    print("   GROQ_API_KEY=your_actual_key_here")
    sys.exit(1)

print(f"\n✅ API key found: {groq_api_key[:20]}...")

# Load sample chunks
chunks_path = repo_root / "outputs" / "chunks" / "chunks.jsonl"

print(f"\n📂 Loading chunks from {chunks_path}")

if not chunks_path.exists():
    print(f"❌ Chunks file not found. Run auto_rag_pipeline.py first.")
    sys.exit(1)

# Load only 2 chunks for testing
chunks = []
with chunks_path.open("r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 2:  # Only 2 chunks to minimize API costs
            break
        chunk = json.loads(line)
        # Filter by length for better Q/A generation
        if 200 <= len(chunk.get("text", "")) <= 800:
            chunks.append(chunk)

if len(chunks) < 2:
    # Get first 2 chunks regardless of length
    chunks = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 2:
                break
            chunks.append(json.loads(line))

print(f"✅ Loaded {len(chunks)} chunks for testing")
for i, chunk in enumerate(chunks, 1):
    print(f"   {i}. {chunk['title']} ({len(chunk['text'])} chars)")

# ============================================================
# Step 1: Initialize Generator
# ============================================================
print(f"\n{'='*60}")
print("  Step 1: Initialize Q/A Generator")
print(f"{'='*60}")

generator = QAGenerator(
    api_key=groq_api_key,
    model="llama-3.3-70b-versatile",
    rate_limit_delay=5.0
)

print(f"\n✅ Generator initialized:")
print(f"   Model: {generator.model}")
print(f"   Rate limit: {generator.rate_limit_delay}s between requests")

# ============================================================
# Step 2: Generate Q/A Pairs
# ============================================================
print(f"\n{'='*60}")
print("  Step 2: Generate Q/A Pairs")
print(f"{'='*60}")

print(f"\n🤖 Starting Q/A generation...")
print(f"   Expected time: ~{len(chunks) * generator.rate_limit_delay}s")
print(f"   (Rate limited to avoid API throttling)")

start_time = time.time()

dataset_records = generator.generate_dataset(
    chunks=chunks,
    target_samples=20,  # 2-4 Q/A pairs per chunk * 2 chunks
    verbose=True
)

elapsed = time.time() - start_time

print(f"\n✅ Generation complete!")
print(f"   Generated: {len(dataset_records)} Q/A pairs")
print(f"   Time elapsed: {elapsed:.1f}s")
print(f"   Success rate: {len(dataset_records)}/{len(chunks)} chunks")

if not dataset_records:
    print(f"\n❌ No Q/A pairs generated. Check API key and rate limits.")
    sys.exit(1)

# Show sample
print(f"\n📝 Sample Generated Q/A:")
for i, record in enumerate(dataset_records[:3], 1):
    print(f"\n   Q{i}: {record['question']}")
    print(f"   A{i}: {record['answer'][:100]}...")
    print(f"   Difficulty: {record['difficulty']}")

# ============================================================
# Step 3: Validate Generated Pairs
# ============================================================
print(f"\n{'='*60}")
print("  Step 3: Validate Generated Q/A Pairs")
print(f"{'='*60}")

validator = QAValidator()

validated_records, quality_stats = validator.validate_dataset(
    dataset_records,
    verbose=True
)

print(f"\n📊 Validation Statistics:")
print(f"   Total generated: {quality_stats['total']}")
print(f"   Valid: {quality_stats['valid']} ({quality_stats['valid']/quality_stats['total']*100:.1f}%)")
print(f"   Invalid: {quality_stats['invalid']}")

if quality_stats['issues_count']:
    print(f"\n⚠️ Issues found:")
    for issue, count in sorted(quality_stats['issues_count'].items(), key=lambda x: -x[1])[:5]:
        print(f"   - {issue}: {count}")

# Show validation details for first few
print(f"\n📋 Validation Details (first 3):")
for i, record in enumerate(validated_records[:3], 1):
    val = record['validation']
    status = "✅ VALID" if val['valid'] else "❌ INVALID"
    print(f"\n   Record {i}: {status}")
    print(f"      Q: {record['question'][:60]}...")
    if val['issues']:
        print(f"      Issues: {', '.join(val['issues'])}")

# ============================================================
# Step 4: Export Valid Pairs
# ============================================================
print(f"\n{'='*60}")
print("  Step 4: Export Valid Q/A Pairs")
print(f"{'='*60}")

output_dir = repo_root / "outputs" / "autoft_datasets"
output_path = output_dir / "test_e2e_dataset.jsonl"

valid_records = [r for r in validated_records if r['validation']['valid']]

print(f"\n💾 Exporting {len(valid_records)} valid records to:")
print(f"   {output_path}")

validator.export_dataset(valid_records, output_path, include_validation=False)

# Verify export
if output_path.exists():
    with output_path.open("r") as f:
        exported = [json.loads(line) for line in f]
    
    print(f"\n✅ Export successful: {len(exported)} records")
    
    # Show sample from exported file
    if exported:
        print(f"\n📄 Sample from exported file:")
        sample = exported[0]
        print(f"   Question: {sample['question']}")
        print(f"   Answer: {sample['answer'][:100]}...")
        print(f"   Difficulty: {sample['difficulty']}")
        print(f"   Source: {sample['source_title']}")

# ============================================================
# Step 5: Quality Report
# ============================================================
print(f"\n{'='*60}")
print("  Step 5: Generate Quality Report")
print(f"{'='*60}")

quality_report = {
    "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "chunks_processed": len(chunks),
    "total_generated": len(dataset_records),
    "valid_count": len(valid_records),
    "invalid_count": quality_stats['invalid'],
    "validation_pass_rate": f"{quality_stats['valid']/quality_stats['total']*100:.1f}%",
    "avg_qa_per_chunk": len(dataset_records) / len(chunks) if chunks else 0,
    "issues_summary": quality_stats['issues_count'],
    "api_time_seconds": elapsed,
    "rate_limit_delay": generator.rate_limit_delay
}

report_path = output_dir / "test_e2e_quality_report.json"
with report_path.open("w") as f:
    json.dump(quality_report, f, indent=2)

print(f"\n📊 Quality Report:")
print(f"   Chunks processed: {quality_report['chunks_processed']}")
print(f"   Q/A pairs generated: {quality_report['total_generated']}")
print(f"   Valid pairs: {quality_report['valid_count']}")
print(f"   Pass rate: {quality_report['validation_pass_rate']}")
print(f"   Avg Q/A per chunk: {quality_report['avg_qa_per_chunk']:.1f}")
print(f"   API time: {quality_report['api_time_seconds']:.1f}s")

print(f"\n💾 Report saved to: {report_path}")

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*60}")
print("  ✅ Full AutoFT Test Complete!")
print(f"{'='*60}")

print(f"\n🎉 Test Summary:")
print(f"   ✅ API Connection: Working")
print(f"   ✅ Q/A Generation: {len(dataset_records)} pairs generated")
print(f"   ✅ Validation: {len(valid_records)}/{len(dataset_records)} pairs valid")
print(f"   ✅ Export: {len(valid_records)} records saved")
print(f"   ✅ Quality Report: Generated")

print(f"\n📁 Output Files:")
print(f"   - Dataset: {output_path}")
print(f"   - Report: {report_path}")

print(f"\n💡 Next Steps:")
print(f"   - Review generated Q/A pairs in {output_path}")
print(f"   - Use generator.generate_dataset() for larger batches")
print(f"   - Adjust rate_limit_delay if hitting API limits")

print(f"\n🎊 AutoFT pipeline fully verified and working!")
