"""Offline regression tests. No credentials, network, or benchmark generation."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from src.context import ContextComposer
from src.evaluation import ContextEvaluator
from src.letta.policy_strategist import LettaPolicyStrategist
from src.orchestration import SelfEvolvingLoop


def matches():
    return [dict(id=f"chunk-{i}", values=vector,
                 metadata=dict(doc_id=f"doc-{i}", title="Title", text="Content", tokens=100))
            for i, vector in enumerate(([1., 0.], [0., 1.]))]


class OrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.embedder = Mock(model="fixture-model")
        self.embedder.embed_query.return_value = np.array([2., 0.])
        self.db = Mock()
        self.db.query.side_effect = lambda **kwargs: matches()
        self.strategist = Mock()
        self.strategist.propose_experiment.return_value = self.proposal()
        self.strategist.decide_on_results.return_value = {"decision": "ACCEPT"}
        self.loop = SelfEvolvingLoop(
            {"budget_tokens": 200}, "mock-agent", "mock-index", ["fixed query"],
            Path(self.directory.name), embedder=self.embedder,
            vectordb=self.db, strategist=self.strategist)

    @staticmethod
    def proposal():
        return {"version": "v0.2", "override": {"budget_tokens": 100},
                "rationale": "fixture", "acceptance_rule": "fixture decision"}

    def experiment(self):
        return self.loop.execute_experiment(self.proposal(), self.loop.run_baseline())

    def test_conversion_alignment(self):
        chunks, vectors = self.loop.matches_to_chunks(list(reversed(matches())), 2)
        self.assertEqual(chunks[0]["chunk_id"], "chunk-1")
        np.testing.assert_array_equal(vectors[0], [0., 1.])
        self.assertEqual(chunks[0]["tokens"], 100)

    def test_conversion_rejects_invalid_records(self):
        for field, value in (("values", []), ("values", [float("nan"), 0]),
                             ("values", [0, 0]), ("id", "")):
            with self.subTest(field=field, value=value):
                record = matches()[0]
                record[field] = value
                with self.assertRaises(ValueError):
                    self.loop.matches_to_chunks([record], 2)
        for field, value in (("tokens", -1), ("tokens", 1.5), ("text", ""), ("doc_id", None)):
            record = matches()[0]
            record["metadata"][field] = value
            with self.assertRaises(ValueError):
                self.loop.matches_to_chunks([record], 2)
        with self.assertRaises(ValueError):
            self.loop.matches_to_chunks([matches()[0], matches()[0]], 2)

    def test_budget_and_metric_definitions(self):
        chunks, vectors = self.loop.matches_to_chunks(matches(), 2)
        composer = ContextComposer(chunks, vectors, {"budget_tokens": 100})
        pkg = composer.compose("query", [0, 1], np.array([1., 0.]))
        self.assertEqual(pkg["used_tokens"], 100)
        self.assertEqual(len(pkg["selected"]), 1)
        scores = ContextEvaluator().evaluate(pkg)
        self.assertAlmostEqual(scores["relevance"], 1.)
        self.assertEqual(scores["redundancy"], 1.)
        self.assertAlmostEqual(scores["coverage"], .6 + .4 / 6)
        self.assertEqual(scores["efficiency"], 1.)
        self.assertAlmostEqual(scores["overall"], .35 + .25 + .2 * (.6 + .4 / 6) + .2)
        composer.policy["budget_tokens"] = 99
        self.assertEqual(composer.compose("q", [0, 1], np.array([1., 0.]))["used_tokens"], 0)

    def test_fixed_comparison_and_snapshot(self):
        baseline = self.loop.run_baseline()
        experiment = self.loop.execute_experiment(self.proposal(), baseline)
        self.assertEqual(self.db.query.call_count, 1)
        self.assertEqual(self.embedder.embed_query.call_count, 1)
        self.assertEqual(self.db.query.call_args.kwargs["namespace"], "kb_chunks")
        self.assertTrue(self.db.query.call_args.kwargs["include_values"])
        self.assertAlmostEqual(experiment["deltas"]["delta_overall"], experiment["metrics"]["overall"] - baseline["overall"])
        snapshot = json.loads((self.loop.state_dir / "evaluation_inputs.json").read_text())
        self.assertEqual(snapshot[0]["query"], "fixed query")
        self.assertEqual(snapshot[0]["embeddings"], [[1., 0.], [0., 1.]])
        with self.assertRaises(ValueError):
            self.loop.evaluate_policy({}, top_k=2)

    def test_accept_and_reject(self):
        experiment = self.experiment()
        original = copy.deepcopy(self.loop.current_policy)
        self.loop.update_policy(experiment, {"decision": "REJECT"})
        self.assertEqual(self.loop.current_policy, original)
        self.loop.update_policy(experiment, {"decision": "ACCEPT"})
        self.assertEqual(self.loop.current_version, "v0.2")
        checkpoint = json.loads((self.loop.state_dir / "checkpoint.json").read_text())
        self.assertEqual(checkpoint["policy"], experiment["policy"])
        self.assertEqual(len(checkpoint["history"]), 2)
        self.strategist.update_experiment_memory.assert_called_with(self.loop.experiment_history[-1])

    def test_invalid_decision_does_not_mutate(self):
        experiment = self.experiment()
        with self.assertRaises(ValueError):
            self.loop.update_policy(experiment, {"decision": "MAYBE"})
        self.assertEqual(self.loop.current_version, "v0.1")
        self.assertEqual(self.loop.experiment_history, [])

    def test_checkpoint_failure_does_not_mutate(self):
        experiment = self.experiment()
        self.loop._save_json("checkpoint.json", {"previous": True})
        with patch("src.orchestration.self_evolving_loop.os.replace", side_effect=OSError("disk error")):
            with self.assertRaises(OSError):
                self.loop.update_policy(experiment, {"decision": "ACCEPT"})
        self.assertEqual(self.loop.current_version, "v0.1")
        self.assertEqual(json.loads((self.loop.state_dir / "checkpoint.json").read_text()), {"previous": True})
        self.assertFalse(list(self.loop.state_dir.glob("*.tmp")))

    def test_notification_failure_preserves_checkpoint(self):
        experiment = self.experiment()
        self.strategist.update_experiment_memory.side_effect = RuntimeError("remote failure")
        with self.assertRaises(RuntimeError):
            self.loop.update_policy(experiment, {"decision": "ACCEPT"})
        self.assertEqual(self.loop.current_version, "v0.2")
        self.assertTrue((self.loop.state_dir / "checkpoint.json").exists())

    def test_retrieval_failure_is_retryable(self):
        self.db.query.side_effect = RuntimeError("retrieval failed")
        with self.assertRaises(RuntimeError):
            self.loop.evaluate_policy({})
        self.assertIsNone(self.loop._evaluation_inputs)
        self.db.query.side_effect = lambda **kwargs: matches()
        self.assertEqual(self.loop.evaluate_policy({})["n_queries"], 1)

    def test_empty_retrieval(self):
        self.db.query.side_effect = None
        self.db.query.return_value = []
        # Preserve the evaluator's existing empty-context semantics.
        self.assertEqual(self.loop.evaluate_policy({})["means"]["redundancy"], 1.)

    def test_invalid_proposals(self):
        for proposal in ({"version": "../bad", "override": {"budget_tokens": 10}},
                         {"version": "v0.2", "override": {"unknown": 1}},
                         {"version": "v0.2", "override": {"budget_tokens": -1}},
                         {"version": "v0.2", "override": {"w_novelty": float("nan")}},
                         {"version": "v0.2", "override": {"order_mode": "unused"}}):
            with self.assertRaises(ValueError):
                self.loop.execute_experiment(proposal, {})
        self.db.query.assert_not_called()

    def test_evolution_end_to_end(self):
        result = self.loop.run_evolution_cycle(max_iterations=1)
        self.assertEqual(result["accepted"], 1)
        self.assertEqual(result["iterations"], 1)
        self.assertEqual(len(result["history"]), 1)
        self.assertEqual(self.db.query.call_count, 1)
        self.assertEqual(self.strategist.propose_experiment.call_args.kwargs["base_policy"], {"budget_tokens": 200})

    def test_zero_iterations(self):
        with self.assertRaises(ValueError):
            self.loop.run_evolution_cycle(max_iterations=0)
        self.db.query.assert_not_called()

    def test_snapshot_write_failure_is_retryable(self):
        with patch.object(self.loop, "_save_json", side_effect=OSError("disk error")):
            with self.assertRaises(OSError):
                self.loop.evaluate_policy({})
        self.assertIsNone(self.loop._evaluation_inputs)

    def test_multiple_queries_and_rejected_cycle(self):
        self.loop.eval_queries = ("first", "second")
        self.strategist.decide_on_results.return_value = {"decision": "REJECT"}
        result = self.loop.run_evolution_cycle(max_iterations=1)
        self.assertEqual(result["rejected"], 1)
        self.assertEqual(result["final_version"], "v0.1")
        self.assertEqual(self.db.query.call_count, 2)
        self.assertEqual(self.embedder.embed_query.call_count, 2)

    def test_document_cap_and_redundancy_filter(self):
        chunks, vectors = self.loop.matches_to_chunks(matches(), 2)
        chunks[1]["doc_id"] = chunks[0]["doc_id"]
        composer = ContextComposer(chunks, vectors, {"max_chunks_per_doc": 1})
        pkg = composer.compose("q", [0, 1], np.array([1., 0.]))
        self.assertEqual(len(pkg["selected"]), 1)
        self.assertEqual(pkg["excluded"][0]["reason"], "doc_cap")
        composer = ContextComposer(chunks, np.array([[1., 0.], [1., 0.]]))
        pkg = composer.compose("q", [0, 1], np.array([1., 0.]))
        self.assertEqual(pkg["excluded"][0]["reason"], "redundant")

    def test_letta_uses_current_response(self):
        strategist = LettaPolicyStrategist.__new__(LettaPolicyStrategist)
        strategist.client = Mock()
        strategist.agent_id = "mock-agent"
        response = SimpleNamespace(messages=[SimpleNamespace(
            message_type="assistant_message", content=json.dumps(self.proposal()))])
        strategist.client.agents.messages.create.return_value = response
        self.assertEqual(strategist.propose_experiment({}, [], {"budget_tokens": 200}), self.proposal())
        strategist.client.agents.messages.list.assert_not_called()
        response.messages[0].content = '{"decision":"REJECT"}'
        self.assertEqual(strategist.decide_on_results(self.proposal(), {}), {"decision": "REJECT"})
        with self.assertRaises(ValueError):
            strategist._assistant_text(SimpleNamespace(messages=[]))


if __name__ == "__main__":
    unittest.main()
