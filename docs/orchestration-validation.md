# Orchestration repair audit

Audit date: 17 October 2026. Base: `94b42afb793f0426aee58e38c0c66ee1e4e0a590`, freshly cloned `origin/master`. Initial tree was clean. Work branch: `fix/dce-orchestration-validation`. No service inference, retrieval, upload, or Letta calls were made during validation.

## Confirmed defects and corrections

| Finding | Correction |
| --- | --- |
| Missing `Config.EMBEDDING_MODEL_NAME` | Use existing `DEFAULT_EMBEDDING_MODEL`. |
| Composer constructed without required chunks/vectors | Construct a composer per query/policy with aligned retrieved inputs. |
| Nonexistent `compose_context` and `evaluate_context` calls | Call existing `compose` and `evaluate` interfaces. |
| Wrong retrieval namespace and IDs treated as array indices | Request `kb_chunks`, metadata and values; build local integer indices in returned-match order. |
| Missing conversion/validation of Pinecone results | Require valid IDs, metadata, positive integral token counts and finite nonzero vectors of the query dimension. Reject duplicates and malformed results. |
| Policies independently retrieve and embed queries | Capture a complete persisted snapshot once, reuse it across baseline/candidates, and reject changes to `top_k`. |
| Letta memory method called with incompatible keyword arguments | Pass the single complete experiment record its signature expects. |
| Strategist searches potentially stale/ambiguously ordered message history | Parse assistant text from the current synchronous response. |
| Current policy omitted from proposal prompt; nested deltas omitted from history display | Include current policy and read stored nested deltas. |
| Malformed proposals/decisions could enter the loop | Validate parameter names, finite numeric values, positive integral constraints, safe unused versions and explicit ACCEPT/REJECT. Reject inactive composer parameter experiments. |
| Policy changed before persistence; rejection history persisted only at cycle end | Commit one atomic authoritative checkpoint per decision before local policy mutation and notification. Retain rejected experiments. |
| Zero iterations leaves `iteration` undefined | Reject nonpositive iteration count before baseline. |
| Emoji logging and CLI help fail under Windows cp1252 | Remove orchestration emoji output and use ASCII arrows in pipeline help. |
| Placeholder package URL/email; console entry-point module omitted from wheel | Correct verified repository URL, remove unverified email, retain license-backed DCE Team attribution, package `scripts`. |
| AutoFT imports undeclared `requests` | Declare the direct dependency. |
| README promises production/performance and calls nonexistent `search` | Replace with actual interfaces, architecture, reproducibility instructions and explicit limitations. |

## File-by-file changes

| File | Change |
| --- | --- |
| `src/orchestration/self_evolving_loop.py` | Correct interfaces, optional injected service objects for offline testing, retrieval conversion/snapshot, proposal/decision validation, atomic checkpoint and failure ordering. |
| `src/letta/policy_strategist.py` | Current-response parsing, current-policy prompt context and nested history deltas. |
| `src/__init__.py` | Remove production claim and nonexistent DCEEngine example. |
| `scripts/auto_rag_pipeline.py` | ASCII arrows in help text for Windows console compatibility. |
| `requirements.txt` | Declare requests. Existing broad service/notebook requirements retained; no untested version pins introduced. |
| `setup.py` | Verified URL, placeholder email removal, restrained description and package CLI implementation. |
| `README.md` | Accurate architecture, installation, configuration, reproducibility and limitations. |
| `tests/offline/test_orchestration.py` | Independent unittest regression suite with mocked service boundaries and temporary state. |
| `docs/orchestration-validation.md` | This audit and validation record. |
| `docs/proposed-pull-request.md` | Review-ready proposed PR description. |

## Validation

Environment: Windows, Python 3.12; NumPy 2.3.5, pandas 3.0.1, requests 2.34.2, python-dotenv 1.2.4, setuptools 84.0.0 and wheel 0.48.0. A local venv inherited bundled packages; this is not a clean full-dependency installation.

Passing checks:

- `python -m unittest discover -s tests/offline -v`: 17 regression tests. Covers composition budgets, document caps, redundancy, frozen metric arithmetic, Pinecone alignment and malformed matches, fixed baseline/candidate comparisons, multiple queries, ACCEPT/REJECT cycles, snapshot/checkpoint failures, notification failure, retryable retrieval, empty retrieval, proposal validation, zero iterations and Letta current-response parsing.
- `python -X utf8 tests/test_imports.py`: all four existing smoke-test groups pass. This script is weaker evidence than assertions; its random short chunks select an empty context.
- `python setup.py bdist_wheel`: wheel built and includes `scripts/auto_rag_pipeline.py` and its console entry point.
- `python -m pip install --no-deps --no-build-isolation -e .`: editable install succeeds.
- `dce-pipeline --help`: succeeds without credentials or service calls.
- `python -m compileall -q src scripts tests/offline` and `git diff --check`: pass.
- `git diff --name-only -- notebooks examples benchmarks archive`: empty; historical artifacts unchanged.

Failing checks: none outstanding in the checks above. Earlier attempts failed because requests was missing, sandbox temporary-directory access was denied, and Windows output could not encode emoji/arrows; those issues were resolved or rerun with required filesystem approval. Packaging emits legacy editable-install and license-classifier deprecation warnings.

Unexecuted checks:

- The nine other legacy test/experiment scripts: several invoke paid/live services, write outputs, or require absent `outputs/` fixtures. They were inspected rather than executed. Unrestricted pytest collection is unsafe because some execute at import time; pytest was not installed in this test environment.
- Live Hugging Face/Pinecone/Letta compatibility, index dimensions/model provenance, agent behavior and remote memory writes.
- Full dependency installation/resolution, clean wheel installation, supported-Python-version matrix, and answer-generation benchmarks.

## Methodology impact

The greedy composer, defaults, metric definitions and frozen evaluation weights are unchanged. In particular, `redundancy` remains **non-redundancy**, so higher is better. Empty context still receives non-redundancy 1; this potentially misleading convention was preserved. Context evaluation remains separate from generated-answer evaluation.

Fixed queries, candidates, stored embeddings, and normalized query vectors are reused throughout a loop. This implements the requested controlled comparison and avoids retrieval drift; it is not evidence of improved scores. A candidate accepted by Letta becomes the next baseline on that same snapshot, preserving the existing loop design. No new acceptance threshold, significance test, scoring definition, model migration, or one-parameter enforcement was added. No historical results were recalculated and no new benchmark claims are made. Synthetic test fixtures are software checks, not experimental findings.

Proposal validation refuses inactive `order_mode` and `redundancy_penalty` experiments instead of implementing new semantics. Positive integral composition limits are required for proposed policies. Numeric weights are checked for finiteness without imposing new weight normalization. Letta's natural-language acceptance criterion remains agent interpreted; implementing an enforced statistical/acceptance system would require methodological approval.

## Remaining limitations

- The configured MiniLM model differs from BGE defaults elsewhere. Without index provenance the correct historical model cannot be inferred. Supply a matching embedder or configuration; do not compare incompatible embedding spaces. No index was rebuilt.
- Letta's memory-update implementation is still a console log stub. The checkpoint is the authoritative implemented history. Live SDK response shape has only been represented by mocks.
- There is no automatic resume, concurrent-writer protection or cross-service transaction. Use fresh state directories. If notification fails after a checkpoint, local acceptance remains committed and the exception stops the cycle.
- Snapshot persistence includes retrieved text; storage and access policy belong to the operator. A partial retrieval or failed snapshot write is not published as a usable baseline.
- Token budgets rely on stored chunk estimates, excluding render headers and generated-answer overhead. Existing no-op composer settings remain no-ops.
- Historical documentation outside README may contain stronger claims; it was retained as historical context, not validated certification.

No required scoring or acceptance redesign was necessary for this repair. Any future change to these research definitions needs separate review.
