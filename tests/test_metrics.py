"""Tests for evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import MetricsCalculator


class TestRetrievalMetrics:
    """Tests for retrieval metrics (no LLM required)."""

    def test_precision_at_k_perfect(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert MetricsCalculator.precision_at_k(retrieved, relevant, k=3) == 1.0

    def test_precision_at_k_none_relevant(self):
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b"}
        assert MetricsCalculator.precision_at_k(retrieved, relevant, k=3) == 0.0

    def test_precision_at_k_partial(self):
        retrieved = ["a", "x", "b", "y"]
        relevant = {"a", "b", "c"}
        assert MetricsCalculator.precision_at_k(retrieved, relevant, k=4) == 0.5

    def test_precision_at_k_respects_k(self):
        retrieved = ["a", "x", "b", "y"]
        relevant = {"a", "b"}
        assert MetricsCalculator.precision_at_k(retrieved, relevant, k=2) == 0.5

    def test_recall_at_k_perfect(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b"}
        assert MetricsCalculator.recall_at_k(retrieved, relevant, k=3) == 1.0

    def test_recall_at_k_partial(self):
        retrieved = ["a", "x", "y"]
        relevant = {"a", "b", "c"}
        assert MetricsCalculator.recall_at_k(retrieved, relevant, k=3) == pytest.approx(1 / 3)

    def test_recall_at_k_empty_relevant(self):
        retrieved = ["a", "b"]
        relevant = set()
        assert MetricsCalculator.recall_at_k(retrieved, relevant, k=2) == 0.0

    def test_f1_at_k(self):
        p, r = 0.5, 0.5
        assert MetricsCalculator.f1_at_k(p, r) == 0.5

    def test_f1_at_k_zero(self):
        assert MetricsCalculator.f1_at_k(0.0, 0.0) == 0.0

    def test_mrr_first_hit(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a"}
        assert MetricsCalculator.mean_reciprocal_rank(retrieved, relevant) == 1.0

    def test_mrr_second_hit(self):
        retrieved = ["x", "a", "b"]
        relevant = {"a"}
        assert MetricsCalculator.mean_reciprocal_rank(retrieved, relevant) == 0.5

    def test_mrr_no_hit(self):
        retrieved = ["x", "y", "z"]
        relevant = {"a"}
        assert MetricsCalculator.mean_reciprocal_rank(retrieved, relevant) == 0.0


class TestAgenticMetrics:
    """Tests for agentic-specific metrics."""

    def test_link_utilization_all_followed(self):
        followed = ["a", "b", "c"]
        available = ["a", "b", "c"]
        assert MetricsCalculator.link_utilization(followed, available) == 1.0

    def test_link_utilization_none_followed(self):
        followed = []
        available = ["a", "b"]
        assert MetricsCalculator.link_utilization(followed, available) == 0.0

    def test_link_utilization_no_available(self):
        assert MetricsCalculator.link_utilization([], []) == 0.0

    def test_link_utilization_partial(self):
        followed = ["a"]
        available = ["a", "b", "c", "d"]
        assert MetricsCalculator.link_utilization(followed, available) == 0.25

    def test_citation_accuracy_all_valid(self):
        answer = "According to [Document doc_1] and [Document doc_2], the answer is yes."
        source_ids = {"doc_1", "doc_2"}
        result = MetricsCalculator.citation_accuracy(answer, source_ids)
        assert result["citation_accuracy"] == 1.0
        assert result["total_citations"] == 2

    def test_citation_accuracy_invalid_citations(self):
        answer = "According to [Document fake_doc], the answer is yes."
        source_ids = {"real_doc"}
        result = MetricsCalculator.citation_accuracy(answer, source_ids)
        assert result["citation_accuracy"] == 0.0
        assert "fake_doc" in result["invalid_citations"]

    def test_citation_accuracy_no_citations(self):
        answer = "The answer is simply yes."
        source_ids = {"doc_1"}
        result = MetricsCalculator.citation_accuracy(answer, source_ids)
        assert result["total_citations"] == 0
