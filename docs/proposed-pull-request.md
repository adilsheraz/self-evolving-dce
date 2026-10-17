# Repair DCE orchestration interfaces and validate controlled policy comparisons

The evolution loop could not run against the repository's component interfaces: it referenced a missing config attribute, constructed the composer without data, called nonexistent composition/evaluation methods, queried the wrong Pinecone namespace, and passed incompatible arguments to Letta memory logging.

This change wires the existing components together, converts Pinecone matches into validated aligned chunks/vectors, and evaluates all policies on one persisted retrieval snapshot. It validates proposals and ACCEPT/REJECT responses, saves an atomic local policy/history checkpoint for each decision, and parses Letta's current response instead of searching message history. It also corrects package metadata, declares requests, includes the CLI module in distributions, fixes Windows console failures, and documents actual capabilities and limitations.

Validation: 21 offline regressions and two installed-SDK response contract tests pass, including against a clean installed wheel in isolated Python mode. Full declared dependency installation and `pip check` succeed. All four existing import smoke groups, compilation, wheel build, editable install, CLI help and diff whitespace checks pass. Follow-up review caught and fixed Letta text-part content compatibility. External services and the nine remaining legacy experiment scripts were not run: no HF/Pinecone/Letta credentials are configured in this workspace. See `docs/orchestration-validation.md` for the file report and failure-handling details.

Research impact: no changes to composer scoring, metric definitions/weights, acceptance ownership, or historical datasets/results. Fixed queries and retrieval inputs prevent comparison drift as required. No new performance or answer-quality claims. Letta memory-block persistence remains unimplemented, SDK behavior is mocked, and embedding-model/index provenance must be verified before live experiments.

Base branch: `master`. Head branch: `fix/dce-orchestration-validation`. No merge or direct master modification is intended.
