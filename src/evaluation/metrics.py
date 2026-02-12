"""Evaluation metrics for the Structured LD Agentic RAG experiments.

Implements answer-level and retrieval-level metrics:
  - Answer Accuracy (LLM-as-judge)
  - Answer Completeness (fact coverage)
  - Retrieval Precision@K / Recall@K
  - Factual Grounding
  - Link Utilization (agentic only)
"""

from __future__ import annotations

import json
import logging
import re

from config.settings import GCP_PROJECT_ID, GCP_REGION, JUDGE_MODEL

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM-as-Judge prompts
# ---------------------------------------------------------------------------

ACCURACY_PROMPT = """\
You are an evaluation judge. Given a question, a ground-truth answer, and a
candidate answer produced by a system, rate the candidate's factual accuracy.

Question: {question}

Ground Truth Answer: {ground_truth}

Candidate Answer: {candidate}

Rate the accuracy on a scale of 1-5:
  1 = Completely wrong or irrelevant
  2 = Mostly wrong, with minor correct elements
  3 = Partially correct, missing key facts
  4 = Mostly correct, minor inaccuracies
  5 = Fully correct and accurate

Respond in JSON format ONLY:
{{"score": <1-5>, "reasoning": "brief explanation"}}
"""

COMPLETENESS_PROMPT = """\
You are an evaluation judge. Given a question, a ground-truth answer containing
key facts, and a candidate answer, rate the completeness of the candidate.

Question: {question}

Ground Truth (contains key facts the answer should cover):
{ground_truth}

Candidate Answer: {candidate}

Rate completeness on a scale of 1-5:
  1 = Covers none of the key facts
  2 = Covers ~25% of key facts
  3 = Covers ~50% of key facts
  4 = Covers ~75% of key facts
  5 = Covers all key facts

Respond in JSON format ONLY:
{{"score": <1-5>, "facts_covered": ["fact1", "fact2"], "facts_missing": ["fact3"], "reasoning": "brief explanation"}}
"""

GROUNDING_PROMPT = """\
You are an evaluation judge. Given a candidate answer and the source documents
it was based on, determine what fraction of claims in the answer are traceable
to the source documents.

Candidate Answer: {candidate}

Source Documents:
{sources}

For each claim in the answer, determine if it is supported by the source documents.

Respond in JSON format ONLY:
{{"grounding_score": <0.0-1.0>, "total_claims": <int>, "grounded_claims": <int>, "ungrounded_claims": ["claim1", "claim2"]}}
"""


class MetricsCalculator:
    """Calculates evaluation metrics for RAG results."""

    def __init__(self, judge_model: str = JUDGE_MODEL) -> None:
        self.judge_model = judge_model
        self._genai_client = None

    @property
    def genai_client(self):
        if self._genai_client is None:
            from google import genai
            import os

            api_key = os.getenv("GOOGLE_CLOUD_API_KEY", "")
            if api_key and "gemini-3" in self.judge_model:
                # gemini-3 models: use Google AI (not Vertex) with API key
                self._genai_client = genai.Client(api_key=api_key)
            else:
                self._genai_client = genai.Client(
                    vertexai=True,
                    project=GCP_PROJECT_ID,
                    location=GCP_REGION,
                )
        return self._genai_client

    # ------------------------------------------------------------------
    # LLM-as-Judge metrics
    # ------------------------------------------------------------------

    def score_accuracy(
        self, question: str, ground_truth: str, candidate: str
    ) -> dict:
        """Score answer accuracy using LLM-as-judge.

        Returns:
            Dict with 'score' (1-5) and 'reasoning'.
        """
        prompt = ACCURACY_PROMPT.format(
            question=question,
            ground_truth=ground_truth,
            candidate=candidate,
        )
        return self._call_judge(prompt, default={"score": 0, "reasoning": "Judge failed"})

    def score_completeness(
        self, question: str, ground_truth: str, candidate: str
    ) -> dict:
        """Score answer completeness using LLM-as-judge.

        Returns:
            Dict with 'score' (1-5), 'facts_covered', 'facts_missing', 'reasoning'.
        """
        prompt = COMPLETENESS_PROMPT.format(
            question=question,
            ground_truth=ground_truth,
            candidate=candidate,
        )
        return self._call_judge(
            prompt,
            default={
                "score": 0,
                "facts_covered": [],
                "facts_missing": [],
                "reasoning": "Judge failed",
            },
        )

    def score_grounding(self, candidate: str, source_documents: list[dict]) -> dict:
        """Score factual grounding of an answer against source documents.

        Returns:
            Dict with 'grounding_score', 'total_claims', 'grounded_claims'.
        """
        sources_text = "\n\n---\n\n".join(
            f"[{d.get('id', i)}]: {d.get('content', '')[:1000]}"
            for i, d in enumerate(source_documents)
        )
        prompt = GROUNDING_PROMPT.format(
            candidate=candidate,
            sources=sources_text,
        )
        return self._call_judge(
            prompt,
            default={
                "grounding_score": 0.0,
                "total_claims": 0,
                "grounded_claims": 0,
                "ungrounded_claims": [],
            },
        )

    def _call_judge(self, prompt: str, default: dict, _retries: int = 1) -> dict:
        """Call the LLM judge and parse JSON response."""
        try:
            from google.genai import types

            response = self.genai_client.models.generate_content(
                model=self.judge_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=1024,
                    response_mime_type="application/json",
                ),
            )
            # Handle thinking models where response.text may be None
            text = response.text
            if text is None and response.candidates:
                text = ""
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "text") and part.text:
                        text += part.text
            if not text:
                logger.warning("Judge returned empty response")
                return default
            text = text.strip()
            # Extract JSON from possible markdown code block or thinking output
            text = text.removeprefix("```json").removesuffix("```").strip()

            # Try direct parse first
            parsed = self._try_parse_json(text)
            if parsed is not None:
                return parsed

            # Try to find and extract a JSON object (handles thinking model preamble)
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                parsed = self._try_parse_json(json_match.group(0))
                if parsed is not None:
                    return parsed

            # If we have retries left, try again
            if _retries > 0:
                import time
                time.sleep(0.5)
                return self._call_judge(prompt, default, _retries=_retries - 1)

            logger.warning("Judge call failed after retries: could not parse JSON")
            return default
        except (json.JSONDecodeError, AttributeError, Exception) as exc:
            if _retries > 0:
                import time
                time.sleep(0.5)
                return self._call_judge(prompt, default, _retries=_retries - 1)
            logger.warning("Judge call failed: %s", exc)
            return default

    @staticmethod
    def _try_parse_json(text: str) -> dict | None:
        """Try to parse JSON, with repair for common truncation issues."""
        # Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Repair: fix unterminated strings by closing open quotes
        # then adding missing closing braces/brackets
        repaired = text.rstrip()

        # Count unmatched braces and brackets
        in_string = False
        escape_next = False
        brace_depth = 0
        bracket_depth = 0

        for ch in repaired:
            if escape_next:
                escape_next = False
                continue
            if ch == '\\':
                escape_next = True
                continue
            if ch == '"' and not escape_next:
                in_string = not in_string
                continue
            if not in_string:
                if ch == '{':
                    brace_depth += 1
                elif ch == '}':
                    brace_depth -= 1
                elif ch == '[':
                    bracket_depth += 1
                elif ch == ']':
                    bracket_depth -= 1

        # If we're still in a string, close it
        if in_string:
            repaired += '"'

        # Close any open brackets/braces
        repaired += ']' * max(bracket_depth, 0)
        repaired += '}' * max(brace_depth, 0)

        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

        # Last resort: try to extract score with regex
        score_match = re.search(r'"score"\s*:\s*(\d+)', text)
        if score_match:
            score = int(score_match.group(1))
            reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)', text)
            reasoning = reasoning_match.group(1) if reasoning_match else "Extracted from partial JSON"
            result = {"score": score, "reasoning": reasoning}
            # Also try to extract facts_covered/facts_missing if present
            facts_match = re.search(r'"facts_covered"\s*:\s*\[([^\]]*)\]', text)
            if facts_match:
                try:
                    result["facts_covered"] = json.loads(f"[{facts_match.group(1)}]")
                except json.JSONDecodeError:
                    result["facts_covered"] = []
                facts_missing_match = re.search(r'"facts_missing"\s*:\s*\[([^\]]*)\]', text)
                if facts_missing_match:
                    try:
                        result["facts_missing"] = json.loads(f"[{facts_missing_match.group(1)}]")
                    except json.JSONDecodeError:
                        result["facts_missing"] = []
            return result

        return None

    # ------------------------------------------------------------------
    # Retrieval metrics (computed directly, no LLM needed)
    # ------------------------------------------------------------------

    @staticmethod
    def precision_at_k(
        retrieved_ids: list[str],
        relevant_ids: set[str],
        k: int = 10,
    ) -> float:
        """Precision@K: fraction of top-K retrieved docs that are relevant."""
        top_k = retrieved_ids[:k]
        if not top_k:
            return 0.0
        relevant_in_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
        return relevant_in_k / len(top_k)

    @staticmethod
    def recall_at_k(
        retrieved_ids: list[str],
        relevant_ids: set[str],
        k: int = 10,
    ) -> float:
        """Recall@K: fraction of relevant docs retrieved in top-K."""
        if not relevant_ids:
            return 0.0
        top_k = retrieved_ids[:k]
        relevant_in_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
        return relevant_in_k / len(relevant_ids)

    @staticmethod
    def f1_at_k(precision: float, recall: float) -> float:
        """F1@K: harmonic mean of precision and recall."""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def mean_reciprocal_rank(
        retrieved_ids: list[str], relevant_ids: set[str]
    ) -> float:
        """MRR: reciprocal of the rank of the first relevant document."""
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_ids:
                return 1.0 / (i + 1)
        return 0.0

    # ------------------------------------------------------------------
    # Agentic metrics
    # ------------------------------------------------------------------

    @staticmethod
    def link_utilization(links_followed: list[str], links_available: list[str]) -> float:
        """Fraction of available links that were followed by the agent."""
        if not links_available:
            return 0.0
        return len(set(links_followed)) / len(set(links_available))

    @staticmethod
    def citation_accuracy(answer: str, source_ids: set[str]) -> dict:
        """Check if citations in the answer correspond to actual sources.

        Looks for patterns like [Document X] or [doc_id] in the answer text.
        """
        # Find all citation-like patterns
        cited = set(re.findall(r'\[(?:Document\s+)?([^\]]+)\]', answer))
        valid_citations = cited & source_ids
        invalid_citations = cited - source_ids

        total = len(cited) if cited else 1
        return {
            "citation_accuracy": len(valid_citations) / total if cited else 0.0,
            "valid_citations": list(valid_citations),
            "invalid_citations": list(invalid_citations),
            "total_citations": len(cited),
        }

    # ------------------------------------------------------------------
    # Aggregate evaluation for a single query result
    # ------------------------------------------------------------------

    def evaluate_result(self, result: dict) -> dict:
        """Compute all applicable metrics for a single query result.

        Args:
            result: Dict with keys: query, answer, ground_truth, retrieved_documents,
                    condition, and optionally: links_followed, links_available, agent_steps.

        Returns:
            Dict of metric name → value.
        """
        metrics = {}

        question = result.get("query", "")
        answer = result.get("answer", "")
        ground_truth = result.get("ground_truth", "")
        retrieved = result.get("retrieved_documents", [])
        condition = result.get("condition", "")

        # LLM-as-judge metrics
        if ground_truth:
            acc = self.score_accuracy(question, ground_truth, answer)
            metrics["accuracy_score"] = acc.get("score", 0)
            metrics["accuracy_reasoning"] = acc.get("reasoning", "")

            comp = self.score_completeness(question, ground_truth, answer)
            metrics["completeness_score"] = comp.get("score", 0)
            metrics["facts_covered"] = comp.get("facts_covered", [])
            metrics["facts_missing"] = comp.get("facts_missing", [])

        # Grounding
        if retrieved:
            grounding = self.score_grounding(answer, retrieved)
            metrics["grounding_score"] = grounding.get("grounding_score", 0.0)
            metrics["total_claims"] = grounding.get("total_claims", 0)
            metrics["grounded_claims"] = grounding.get("grounded_claims", 0)

        # Citation accuracy
        source_ids = {d.get("id", "") for d in retrieved}
        cit = self.citation_accuracy(answer, source_ids)
        metrics.update(cit)

        # Agentic-specific metrics
        if condition in ("C4", "C5", "C6"):
            links_followed = result.get("links_followed", [])
            links_available = result.get("links_available", [])
            metrics["link_utilization"] = self.link_utilization(
                links_followed, links_available
            )
            metrics["links_followed_count"] = len(links_followed)
            metrics["links_available_count"] = len(links_available)
            metrics["max_hop_depth"] = result.get("max_hop_depth", 0)
            metrics["num_tool_calls"] = len(result.get("agent_steps", []))

        return metrics
