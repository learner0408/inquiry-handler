"""DeepEval RAG suite. Run via: deepeval test run evals/test_rag.py (never plain pytest)."""

import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase


@pytest.mark.parametrize(
    "case",
    [
        LLMTestCase(
            input="What are your hours?",
            actual_output="TODO: wire pipeline.handle_inquiry output",
            expected_output="TODO: expected answer",
            retrieval_context=["TODO: retrieved chunk"],
        ),
    ],
)
def test_rag_groundedness(case: LLMTestCase) -> None:
    assert_test(case, [FaithfulnessMetric(threshold=0.8), AnswerRelevancyMetric(threshold=0.7)])
