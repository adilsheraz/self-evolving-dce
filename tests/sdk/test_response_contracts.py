"""Offline contracts against installed service SDK models; no network calls."""
import unittest
from datetime import datetime, timezone
from unittest.mock import Mock

import numpy as np
from letta_client.types.agents.assistant_message import AssistantMessage
from letta_client.types.agents.letta_response import LettaResponse
from pinecone.models.vectors.responses import QueryResponse
from pinecone.models.vectors.vector import ScoredVector

from src.letta.policy_strategist import LettaPolicyStrategist
from src.orchestration import SelfEvolvingLoop
from src.vectordb import PineconeLTM


class ResponseContracts(unittest.TestCase):
    def test_letta_string_and_text_parts(self):
        for content in ('{"decision":"REJECT"}',
                        [{"type": "text", "text": '{"decision":'},
                         {"type": "text", "text": '"REJECT"}'}]):
            message = AssistantMessage(id="fixture", date=datetime.now(timezone.utc),
                                       message_type="assistant_message", content=content)
            response = LettaResponse.model_construct(messages=[message])
            self.assertEqual(LettaPolicyStrategist._assistant_text(response),
                             '{"decision":"REJECT"}')

    def test_pinecone_query_response_conversion(self):
        match = ScoredVector(id="fixture", score=1., values=[1., 0.],
                             metadata={"doc_id": "doc", "title": "Title",
                                       "text": "Text", "tokens": 100.})
        response = QueryResponse(matches=[match], namespace="kb_chunks")
        db = PineconeLTM.__new__(PineconeLTM)
        db.index = Mock()
        db.index.query.return_value = response
        matches = db.query(np.array([1., 0.]), include_values=True)
        chunks, vectors = SelfEvolvingLoop.matches_to_chunks(matches, 2)
        self.assertEqual(chunks[0]["chunk_id"], "fixture")
        np.testing.assert_array_equal(vectors, [[1., 0.]])


if __name__ == "__main__":
    unittest.main()
