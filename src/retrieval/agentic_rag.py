"""Agentic RAG pipeline — uses Google ADK for multi-hop retrieval.

Unlike standard RAG, the agent can:
  1. Search the document index (like standard RAG)
  2. Follow linked entity URIs via content negotiation
  3. Query a neural search endpoint across the knowledge graph

This enables multi-hop reasoning: retrieve → discover links → follow → aggregate → answer.
Used for conditions C4, C5, and C6.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field

import httpx

from config.settings import (
    ACCEPT_JSONLD,
    AGENT_TIMEOUT_SECONDS,
    GCP_PROJECT_ID,
    GCP_REGION,
    GENERATION_MODEL,
    MAX_LINKS_PER_HOP,
    MAX_TRAVERSAL_HOPS,
    NEURAL_SEARCH_ENDPOINT,
    TOP_K,
    WORDLIFT_MCP_ENDPOINT,
    get_wordlift_key,
)
from src.indexing.vectorsearch import VectorSearchManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class AgentStep:
    """A single step in the agent's reasoning trace."""

    tool: str
    input: dict
    output: str
    hop_depth: int = 0


@dataclass
class AgenticRAGResult:
    """Result from an agentic RAG query."""

    query: str
    answer: str
    retrieved_documents: list[dict] = field(default_factory=list)
    agent_steps: list[AgentStep] = field(default_factory=list)
    links_followed: list[str] = field(default_factory=list)
    links_available: list[str] = field(default_factory=list)
    max_hop_depth: int = 0
    generation_model: str = ""
    condition: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def link_utilization(self) -> float:
        """Fraction of available links that were actually followed."""
        if not self.links_available:
            return 0.0
        return len(self.links_followed) / len(self.links_available)


# ---------------------------------------------------------------------------
# Agent tools — these are the capabilities the ADK agent can invoke
# ---------------------------------------------------------------------------


class AgentTools:
    """Tools available to the agentic RAG agent."""

    def __init__(
        self,
        vector_search: VectorSearchManager,
        condition: str,
        domain_account_id: str = "",
        top_k: int = TOP_K,
        max_hops: int = MAX_TRAVERSAL_HOPS,
        max_links: int = MAX_LINKS_PER_HOP,
    ) -> None:
        self.vector_search = vector_search
        self.condition = condition
        self.domain_account_id = domain_account_id
        self.top_k = top_k
        self.max_hops = max_hops
        self.max_links = max_links
        self.http_client = httpx.Client(timeout=30.0, follow_redirects=True)

        # Tracking
        self._steps: list[AgentStep] = []
        self._links_followed: list[str] = []
        self._links_available: list[str] = []
        self._current_hop = 0

    def search_documents(self, query: str) -> str:
        """Search the document index using hybrid search.

        Use this tool to find documents relevant to the user's question.
        Returns the top matching documents with their content.

        Args:
            query: Natural language search query.

        Returns:
            Formatted string of matching documents.
        """
        logger.info("  [Tool] search_documents: '%s'", query)
        results = self.vector_search.search(
            condition=self.condition,
            query=query,
            top_k=self.top_k,
            hybrid=True,
        )

        # Extract linked entity URLs from results
        for doc in results:
            content = doc.get("content", "")
            urls = self._extract_entity_urls(content)
            self._links_available.extend(urls)

        step = AgentStep(
            tool="search_documents",
            input={"query": query},
            output=f"Found {len(results)} documents",
            hop_depth=0,
        )
        self._steps.append(step)

        # Format results
        parts = []
        for i, doc in enumerate(results):
            doc_id = doc.get("id", f"doc_{i}")
            content = doc.get("content", "")[:2000]
            parts.append(f"[Document {doc_id}]\n{content}")

        return "\n\n---\n\n".join(parts) if parts else "No documents found."

    def follow_entity_link(self, url: str) -> str:
        """Follow a linked entity URI to fetch its data via content negotiation.

        Use this tool when you find a linked entity URL in a document and need
        more information about that entity. The URL will be fetched as JSON-LD
        for structured data.

        Args:
            url: The entity URI to dereference (e.g. https://data.wordlift.io/wl.../...).

        Returns:
            The entity data as formatted text.
        """
        if self._current_hop >= self.max_hops:
            return f"Maximum traversal depth ({self.max_hops}) reached. Cannot follow more links."

        if len(self._links_followed) >= self.max_links * (self._current_hop + 1):
            return f"Maximum links per hop ({self.max_links}) reached."

        logger.info("  [Tool] follow_entity_link: %s (hop %d)", url, self._current_hop + 1)
        self._current_hop += 1
        self._links_followed.append(url)

        try:
            # Try JSON-LD first
            jsonld_url = url
            if not url.endswith(".json"):
                jsonld_url = url.rstrip("/") + ".json"

            resp = self.http_client.get(
                jsonld_url,
                headers={"Accept": ACCEPT_JSONLD},
            )
            resp.raise_for_status()
            data = resp.json()

            step = AgentStep(
                tool="follow_entity_link",
                input={"url": url},
                output=f"Fetched entity data ({len(json.dumps(data))} chars)",
                hop_depth=self._current_hop,
            )
            self._steps.append(step)

            # Extract any further links
            if isinstance(data, list):
                for entity in data:
                    urls = self._extract_entity_urls(json.dumps(entity))
                    self._links_available.extend(urls)

            return json.dumps(data, indent=2, ensure_ascii=False)[:3000]

        except httpx.HTTPError as exc:
            step = AgentStep(
                tool="follow_entity_link",
                input={"url": url},
                output=f"Error: {exc}",
                hop_depth=self._current_hop,
            )
            self._steps.append(step)
            return f"Failed to fetch entity at {url}: {exc}"

    def search_knowledge_graph(self, query: str, entity_type: str = "") -> str:
        """Search the knowledge graph for entities matching a natural language query.

        Use this tool when you need to discover entities that are semantically
        related but not directly linked in the graph. This performs a neural
        search across all entities in the knowledge graph via the WordLift
        GraphQL entitySearch API.

        Args:
            query: Natural language description of what you're looking for.
            entity_type: Optional Schema.org type filter (e.g. "Product", "Organization").

        Returns:
            List of matching entities with their details.
        """
        logger.info("  [Tool] search_knowledge_graph: '%s' (type=%s)", query, entity_type)

        # Get the API key for this domain's knowledge graph
        api_key = get_wordlift_key(self.domain_account_id) if self.domain_account_id else ""

        if not api_key:
            step = AgentStep(
                tool="search_knowledge_graph",
                input={"query": query, "entity_type": entity_type},
                output="No API key configured for neural search",
                hop_depth=self._current_hop,
            )
            self._steps.append(step)
            return "Neural search unavailable — no API key configured for this knowledge graph."

        try:
            # Build GraphQL entitySearch query (same schema as WordLift MCP)
            escaped = query.replace("\\", "\\\\").replace('"', '\\"')

            # Type constraint
            type_constraint = ""
            if entity_type:
                type_constraint = f'typeConstraint: {{ in: ["http://schema.org/{entity_type}"] }}'

            query_parts = [f'search: {{ string: "{escaped}" }}']
            if type_constraint:
                query_parts.append(type_constraint)
            query_block = ", ".join(query_parts)

            graphql_query = f"""{{
  entitySearch(
    page: 0
    rows: 10
    query: {{ {query_block} }}
  ) {{
    iri
    name: string(name: "schema:name")
    types: refs(name: "rdf:type")
    url: string(name: "schema:url")
    description: string(name: "schema:description")
    score: float(name: "_:score")
  }}
}}"""

            resp = self.http_client.post(
                "https://api.wordlift.io/graphql",
                json={"query": graphql_query},
                headers={
                    "Authorization": f"Key {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            result_data = resp.json()

            entities_found = []
            for entity in result_data.get("data", {}).get("entitySearch", []) or []:
                parts = [f"URI: {entity.get('iri', '')}"]
                name = entity.get("name")
                if name:
                    parts.append(f"Name: {name}")
                types = entity.get("types")
                if types:
                    parts.append(f"Types: {', '.join(types)}")
                desc = entity.get("description")
                if desc:
                    parts.append(f"Description: {desc}")
                url = entity.get("url")
                if url:
                    parts.append(f"URL: {url}")
                score = entity.get("score")
                if score:
                    parts.append(f"Score: {score:.3f}")
                entities_found.append(" | ".join(parts))

            output_text = "\n\n".join(entities_found) if entities_found else "No entities found."

            step = AgentStep(
                tool="search_knowledge_graph",
                input={"query": query, "entity_type": entity_type},
                output=f"GraphQL entitySearch returned {len(entities_found)} results",
                hop_depth=self._current_hop,
            )
            self._steps.append(step)

            # Extract any entity URLs from results for tracking
            urls = self._extract_entity_urls(output_text)
            self._links_available.extend(urls)

            return output_text[:3000]

        except httpx.HTTPError as exc:
            step = AgentStep(
                tool="search_knowledge_graph",
                input={"query": query, "entity_type": entity_type},
                output=f"GraphQL error: {exc}",
                hop_depth=self._current_hop,
            )
            self._steps.append(step)
            return f"Neural search failed: {exc}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_entity_urls(text: str) -> list[str]:
        """Extract entity URLs from text content (links in JSON-LD or HTML).

        Uses ENTITY_URL_PATTERNS from settings to match URLs from all
        known Linked Data endpoints (data.wordlift.io, open.salzburgerland.com, etc.).
        """
        import re

        from config.settings import ENTITY_URL_PATTERNS

        urls = set()
        for pattern in ENTITY_URL_PATTERNS:
            for match in re.finditer(pattern, text):
                url = match.group(0).rstrip(".,;")
                urls.add(url)
        return list(urls)

    def get_tracking_data(self) -> dict:
        """Return tracking data for evaluation."""
        return {
            "steps": [
                {"tool": s.tool, "hop_depth": s.hop_depth, "output_preview": s.output[:100]}
                for s in self._steps
            ],
            "links_followed": self._links_followed,
            "links_available": list(set(self._links_available)),
            "max_hop_depth": self._current_hop,
        }

    def reset(self) -> None:
        """Reset tracking state for a new query."""
        self._steps.clear()
        self._links_followed.clear()
        self._links_available.clear()
        self._current_hop = 0

    def close(self) -> None:
        self.http_client.close()


# ---------------------------------------------------------------------------
# Agentic RAG pipeline
# ---------------------------------------------------------------------------

AGENT_INSTRUCTION = """\
You are a research assistant with access to a knowledge graph and document search.
Your goal is to answer the user's question as accurately and completely as possible.

STRATEGY:
1. First, use search_documents to find relevant documents.
2. Examine the results for linked entity URLs (especially data.wordlift.io URLs).
3. If the question requires information about related entities (provider, offers, etc.),
   use follow_entity_link to fetch their data.
4. If you need to discover entities not directly linked, use search_knowledge_graph.
5. Synthesize all gathered information into a comprehensive answer.
6. Cite your sources by document ID or entity URL.

RULES:
- Only state facts you found in the retrieved data. Do not hallucinate.
- Follow at most {max_hops} link hops.
- If you cannot find sufficient information, say so explicitly.
"""


class AgenticRAGPipeline:
    """Agentic RAG pipeline using Google ADK for multi-hop retrieval."""

    def __init__(
        self,
        vector_search: VectorSearchManager | None = None,
        model: str = GENERATION_MODEL,
        top_k: int = TOP_K,
        max_hops: int = MAX_TRAVERSAL_HOPS,
        max_links: int = MAX_LINKS_PER_HOP,
        domain_account_id: str = "",
    ) -> None:
        self.vector_search = vector_search or VectorSearchManager()
        self.model = model
        self.top_k = top_k
        self.max_hops = max_hops
        self.max_links = max_links
        self.domain_account_id = domain_account_id
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

    def query(self, question: str, condition: str) -> AgenticRAGResult:
        """Run the agentic RAG pipeline for a single query.

        This uses a tool-calling loop: the LLM decides which tools to invoke,
        follows links, and aggregates information before answering.

        Args:
            question: Natural language query.
            condition: Experimental condition (C4, C5, or C6).

        Returns:
            AgenticRAGResult with answer, agent trace, and metrics.
        """
        logger.info("Agentic query: '%s' (condition=%s)", question, condition)

        # Create agent tools
        tools = AgentTools(
            vector_search=self.vector_search,
            condition=condition,
            domain_account_id=self.domain_account_id,
            top_k=self.top_k,
            max_hops=self.max_hops,
            max_links=self.max_links,
        )

        try:
            answer = self._run_agent(question, tools)
        finally:
            tracking = tools.get_tracking_data()
            tools.close()

        return AgenticRAGResult(
            query=question,
            answer=answer,
            agent_steps=[
                AgentStep(
                    tool=s["tool"],
                    input={},
                    output=s["output_preview"],
                    hop_depth=s["hop_depth"],
                )
                for s in tracking["steps"]
            ],
            links_followed=tracking["links_followed"],
            links_available=tracking["links_available"],
            max_hop_depth=tracking["max_hop_depth"],
            generation_model=self.model,
            condition=condition,
            metadata={
                "top_k": self.top_k,
                "max_hops": self.max_hops,
                "retrieval_mode": "agentic",
                "num_tool_calls": len(tracking["steps"]),
            },
        )

    def _run_agent(self, question: str, tools: AgentTools) -> str:
        """Run the ADK agent loop.

        NOTE: This is a simplified version. The full ADK integration would use:
            from google.adk import Agent
            agent = Agent(model=self.model, tools=[...], instruction=...)
            response = agent.run(question)

        For now, we implement a manual tool-calling loop with Gemini.
        """
        instruction = AGENT_INSTRUCTION.format(max_hops=self.max_hops)

        # Define tools as function declarations for Gemini
        tool_declarations = [
            {
                "name": "search_documents",
                "description": tools.search_documents.__doc__,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "follow_entity_link",
                "description": tools.follow_entity_link.__doc__,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "Entity URL to fetch"},
                    },
                    "required": ["url"],
                },
            },
            {
                "name": "search_knowledge_graph",
                "description": tools.search_knowledge_graph.__doc__,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Neural search query"},
                        "entity_type": {
                            "type": "string",
                            "description": "Optional Schema.org type filter",
                        },
                    },
                    "required": ["query"],
                },
            },
        ]

        # Tool dispatch map
        tool_functions = {
            "search_documents": tools.search_documents,
            "follow_entity_link": tools.follow_entity_link,
            "search_knowledge_graph": tools.search_knowledge_graph,
        }

        # Conversation history for the agent loop
        messages = [
            {"role": "user", "parts": [{"text": f"{instruction}\n\nQuestion: {question}"}]}
        ]

        max_iterations = 5 + self.max_hops * 2
        for iteration in range(max_iterations):
            logger.info("  Agent iteration %d", iteration + 1)

            try:
                response = self.genai_client.models.generate_content(
                    model=self.model,
                    contents=messages,
                    config={
                        "tools": [{"function_declarations": tool_declarations}],
                    },
                )
            except Exception as exc:
                logger.error("Agent generation error: %s", exc)
                return f"Agent error: {exc}"

            # Check if the model wants to call tools
            candidate = response.candidates[0] if response.candidates else None
            if not candidate:
                return "No response from agent."

            has_function_call = False
            function_responses = []

            for part in candidate.content.parts:
                if hasattr(part, "function_call") and part.function_call:
                    has_function_call = True
                    fc = part.function_call
                    fn_name = fc.name
                    fn_args = dict(fc.args) if fc.args else {}

                    logger.info("  Agent calls: %s(%s)", fn_name, fn_args)

                    # Execute the tool
                    fn = tool_functions.get(fn_name)
                    if fn:
                        result = fn(**fn_args)
                    else:
                        result = f"Unknown tool: {fn_name}"

                    function_responses.append({
                        "name": fn_name,
                        "response": {"result": result},
                    })

            if has_function_call:
                # Add model response and tool results to conversation
                messages.append({"role": "model", "parts": candidate.content.parts})
                messages.append({
                    "role": "user",
                    "parts": [
                        {"function_response": fr}
                        for fr in function_responses
                    ],
                })
            else:
                # Model produced a final text answer
                # Handle thinking models where response.text may be None
                text = response.text
                if text is None:
                    text = ""
                    for p in candidate.content.parts:
                        if hasattr(p, "text") and p.text:
                            text += p.text
                return text.strip()

        return "Agent reached maximum iterations without producing a final answer."

    def batch_query(
        self, queries: list[dict], condition: str
    ) -> list[AgenticRAGResult]:
        """Run agentic RAG for multiple queries."""
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
    parser = argparse.ArgumentParser(description="Agentic RAG pipeline")
    parser.add_argument("--query", type=str, help="Single query to test")
    parser.add_argument("--condition", type=str, default="C6", help="Condition (C4-C6)")
    parser.add_argument("--test", action="store_true", help="Run test query")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    pipeline = AgenticRAGPipeline()

    if args.test:
        q = args.query or "Who provides Student Accident Insurance and what else do they offer?"
        result = pipeline.query(q, args.condition)
        print(f"\nQuery: {result.query}")
        print(f"Condition: {result.condition}")
        print(f"Agent steps: {len(result.agent_steps)}")
        print(f"Links followed: {len(result.links_followed)} / {len(result.links_available)}")
        print(f"Max hop depth: {result.max_hop_depth}")
        print(f"Link utilization: {result.link_utilization:.1%}")
        print(f"\nAnswer:\n{result.answer}")

    elif args.query:
        result = pipeline.query(args.query, args.condition)
        print(json.dumps({
            "query": result.query,
            "answer": result.answer,
            "agent_steps": len(result.agent_steps),
            "links_followed": result.links_followed,
            "link_utilization": result.link_utilization,
            "max_hop_depth": result.max_hop_depth,
        }, indent=2))


if __name__ == "__main__":
    main()
