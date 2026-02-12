"""Standard RAG pipeline — query → retrieve → generate.

Uses Vertex AI Vector Search 2.0 for retrieval and Gemini for
answer generation. No agentic link-following; this is the
baseline retrieval mode for conditions C1–C3.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field

from config.settings import GCP_PROJECT_ID, GCP_REGION, GENERATION_MODEL, TOP_K
from src.indexing.vectorsearch import VectorSearchManager

logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    """Result from a single RAG query."""

    query: str
    answer: str
    retrieved_documents: list[dict] = field(default_factory=list)
    generation_model: str = ""
    condition: str = ""
    metadata: dict = field(default_factory=dict)


ANSWER_PROMPT = """\
You are a helpful assistant answering questions based on the provided context documents.

IMPORTANT RULES:
1. Answer ONLY based on the information in the context documents below.
2. If the context does not contain enough information, say "I cannot find sufficient information to answer this question."
3. Cite the specific documents you used by their ID.
4. Be precise and factual.

CONTEXT DOCUMENTS:
{context}

QUESTION: {query}

Provide a clear, accurate answer with citations to the source documents.
"""


class StandardRAGPipeline:
    """Standard RAG: retrieve relevant docs then generate an answer."""

    def __init__(
        self,
        vector_search: VectorSearchManager | None = None,
        model: str = GENERATION_MODEL,
        top_k: int = TOP_K,
    ) -> None:
        self.vector_search = vector_search or VectorSearchManager()
        self.model = model
        self.top_k = top_k
        self._genai_client = None

    @property
    def genai_client(self):
        if self._genai_client is None:
            from google import genai

            self._genai_client = genai.Client(
                vertexai=True,
                project=GCP_PROJECT_ID,
                location=GCP_REGION,
            )
        return self._genai_client

    def query(self, question: str, condition: str) -> RAGResult:
        """Run the full RAG pipeline for a single query.

        Args:
            question: Natural language query.
            condition: Experimental condition (C1, C2, or C3).

        Returns:
            RAGResult with answer, retrieved docs, and metadata.
        """
        # Step 1: Retrieve
        logger.info("Retrieving for: '%s' (condition=%s)", question, condition)
        retrieved = self.vector_search.search(
            condition=condition,
            query=question,
            top_k=self.top_k,
            hybrid=True,
        )
        logger.info("  Retrieved %d documents", len(retrieved))

        # Step 2: Build context
        context = self._build_context(retrieved)

        # Step 3: Generate answer
        answer = self._generate_answer(question, context)

        return RAGResult(
            query=question,
            answer=answer,
            retrieved_documents=retrieved,
            generation_model=self.model,
            condition=condition,
            metadata={
                "top_k": self.top_k,
                "num_retrieved": len(retrieved),
                "retrieval_mode": "standard",
            },
        )

    def _build_context(self, documents: list[dict]) -> str:
        """Format retrieved documents into a context string for the LLM."""
        parts = []
        for i, doc in enumerate(documents):
            doc_id = doc.get("id", f"doc_{i}")
            content = doc.get("content", "")
            # Truncate very long documents
            if len(content) > 3000:
                content = content[:3000] + "\n... [truncated]"
            parts.append(f"[Document {doc_id}]\n{content}\n")
        return "\n---\n".join(parts)

    def _generate_answer(self, query: str, context: str) -> str:
        """Generate an answer using Gemini given the context."""
        prompt = ANSWER_PROMPT.format(context=context, query=query)

        try:
            response = self.genai_client.models.generate_content(
                model=self.model,
                contents=prompt,
            )
            text = response.text
            if text is None and response.candidates:
                text = ""
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "text") and part.text:
                        text += part.text
            return (text or "").strip()
        except Exception as exc:
            logger.error("Generation failed: %s", exc)
            return f"Error generating answer: {exc}"

    def batch_query(
        self, queries: list[dict], condition: str
    ) -> list[RAGResult]:
        """Run RAG pipeline for multiple queries.

        Args:
            queries: List of query dicts with at least a 'query' field.
            condition: Experimental condition.

        Returns:
            List of RAGResult objects.
        """
        results = []
        for i, q in enumerate(queries):
            logger.info("Query %d/%d", i + 1, len(queries))
            result = self.query(q["query"], condition)
            result.metadata["query_type"] = q.get("type", "unknown")
            result.metadata["ground_truth"] = q.get("ground_truth", "")
            results.append(result)
        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Standard RAG pipeline")
    parser.add_argument("--query", type=str, help="Single query to test")
    parser.add_argument("--condition", type=str, default="C2", help="Condition (C1-C3)")
    parser.add_argument("--test", action="store_true", help="Run test query")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    pipeline = StandardRAGPipeline()

    if args.test:
        q = args.query or "What is Student Accident Insurance?"
        result = pipeline.query(q, args.condition)
        print(f"\nQuery: {result.query}")
        print(f"Condition: {result.condition}")
        print(f"Retrieved: {len(result.retrieved_documents)} documents")
        print(f"\nAnswer:\n{result.answer}")

    elif args.query:
        result = pipeline.query(args.query, args.condition)
        print(json.dumps({
            "query": result.query,
            "answer": result.answer,
            "num_retrieved": len(result.retrieved_documents),
            "condition": result.condition,
        }, indent=2))


if __name__ == "__main__":
    main()
