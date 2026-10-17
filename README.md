# Self-Evolving Dynamic Context Engine (DCE)

A research implementation of budget-aware context composition and Letta-guided policy experiments. Offline tests validate interfaces and control flow; they do not establish production readiness, autonomous reliability, or answer-quality improvements.

## Architecture

`ingestion → chunking → embeddings → Pinecone kb_chunks → ContextComposer → ContextEvaluator`

`SelfEvolvingLoop` coordinates baseline evaluation, a Letta proposal, candidate evaluation, a Letta ACCEPT/REJECT decision, and local persistence. The composer greedily balances retrieval similarity, novelty, and document diversity under token and document caps. The evaluator measures relevance, non-redundancy, coverage, and budget efficiency. It generates no answers and uses no LLM judge. AutoFT generation and answer benchmarks are separate workflows.

## Installation

Python 3.9+ is declared by the package; this repair was checked on Python 3.12.

```sh
git clone https://github.com/adilsheraz/self-evolving-dce.git
cd self-evolving-dce
python -m venv .venv
# Activate .venv using your shell's activation command.
python -m pip install -e .
```

`requirements.txt` contains runtime and notebook dependencies and is not a reproducibility lockfile. Service SDK compatibility must be verified before live use. The packaged `dce-pipeline --help` command exposes the ingestion/upload pipeline; executing that pipeline can call external services and create a Pinecone index.

For offline repair tests, only NumPy, pandas, requests, and python-dotenv are needed:

```sh
python -m pip install numpy pandas requests python-dotenv
python -m unittest discover -s tests/offline -v
python -X utf8 tests/test_imports.py
```

With the declared service SDKs installed, their response models can also be tested offline:

```sh
python -m unittest discover -s tests/sdk -v
```

Other `tests/test_*.py` files include executable integration/experiment scripts. Several contact services at import time or require files under `outputs/`. Review them and authorize external calls before unrestricted test discovery.

## Configuration and usage

Set `HF_TOKEN`, `PINECONE_API_KEY`, and `LETTA_API_KEY` in your environment or a local `.env`. Never commit credentials. Use an existing Letta agent. `Config.DEFAULT_EMBEDDING_MODEL` controls orchestration query embeddings. **Its model, dimensions, and query prefix must match the embeddings already in the index**; matching dimensions alone is insufficient. BGE defaults elsewhere differ from the configured MiniLM model, so verify how the index was built.

```python
from src import ContextComposer, ContextEvaluator
import numpy as np

# Illustrative local inputs, not benchmark data.
chunks = [{"chunk_id": "a", "doc_id": "doc", "title": "Example",
           "text": "Example content", "tokens": 100}]
composer = ContextComposer(chunks, np.array([[1., 0.]]),
                           policy={"budget_tokens": 200})
package = composer.compose("example query", [0], np.array([1., 0.]))
scores = ContextEvaluator().evaluate(package)
```

With separately approved service access:

```python
from src import SelfEvolvingLoop

loop = SelfEvolvingLoop(
    base_policy={"budget_tokens": 1200},
    letta_agent_id="YOUR_EXISTING_AGENT_ID",
    pinecone_index_name="YOUR_MATCHING_INDEX",
    eval_queries=["YOUR_FIXED_QUERY"],
    state_dir="state/my-run",
)
# Calls Hugging Face, Pinecone and Letta:
# summary = loop.run_evolution_cycle(max_iterations=1)
```

Pinecone matches must contain stored embedding values and metadata `doc_id`, `title`, `text`, and a positive integral `tokens` count. Missing data raises an error rather than silently dropping candidates or re-embedding text. Empty retrieval retains the existing evaluator's empty-context behavior.

## Reproducibility and limitations

Use a new state directory for each run. The first evaluation captures fixed queries, normalized query vectors, retrieved metadata, and stored candidate embeddings in `evaluation_inputs.json`. Every policy in that loop reuses those inputs and the same `top_k`; fresh retrieval requires a new loop/run. To replay offline, load this snapshot, construct `ContextComposer` for each query and policy, and evaluate with `ContextEvaluator`. Record the Git commit, package versions, policies, index provenance, and model separately.

`baseline_metrics.json` stores the initial baseline. Each decision atomically replaces `checkpoint.json` with the current policy and full decision history, including rejected experiments. The completed cycle also writes `evolution_history.json`. Disk-write failure prevents policy mutation. A subsequent Letta notification failure propagates after the local checkpoint commits; inspect the checkpoint before retrying. There is no automatic resume or multi-process coordination. Snapshots contain source text.

Letta remains the acceptance decision-maker. Acceptance rules are natural-language instructions, not a locally enforced statistical test. Validation rejects malformed proposals, unsafe/reused versions, unknown parameters, and experiments on inactive settings. It introduces no new acceptance threshold or experimental design. The existing `order_mode` and `redundancy_penalty` parameters are unused by composition. Token limits use stored estimates and exclude rendering headers and answer-generation overhead.

Letta's `update_experiment_memory()` only prints a log summary; it does not persist an agent memory block. Local checkpoints and history passed in proposal prompts provide the implemented audit trail. Current-response parsing supports strings and typed text parts and is tested with installed SDK response models; live provider behavior has not been validated.

Historical datasets, embeddings, evaluations, and experiment logs remain under `notebooks/mvp_kb_outputs/`. They were not regenerated or reinterpreted. Context metrics are proxies and cannot establish generated-answer accuracy. See [the audit report](docs/orchestration-validation.md) for corrections, test evidence, and methodology impact. Older documents, including `PRODUCTION_SUMMARY.md`, contain historical claims and are not certification of this workflow.

## License and attribution

MIT; see [LICENSE](LICENSE). Author remains DCE Team as recorded in the license. The package URL points to the verified repository; the placeholder email was removed because no verified author email was available.
