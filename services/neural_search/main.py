"""Neural Search API — Cloud Run endpoint for the experiment.

A thin proxy that exposes a unified neural search interface over
knowledge graphs. Queries can be routed to either:
  1. WordLift MCP (mcp.wordlift.io) — the managed neural search
  2. Direct GraphQL queries against the WordLift KG

Deployed on Google Cloud Run for a controlled, loggable endpoint
that the agentic RAG pipeline can call during experiments.

This endpoint is designed to be agent-friendly: it returns structured
JSON results with entity URIs, types, and relevance scores.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, HTTPException, Header, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WORDLIFT_MCP_ENDPOINT = os.getenv("WORDLIFT_MCP_ENDPOINT", "https://mcp.wordlift.io")
GRAPHQL_ENDPOINT = os.getenv("WORDLIFT_GRAPHQL_ENDPOINT", "https://api.wordlift.io/graphql")
LOG_QUERIES = os.getenv("LOG_QUERIES", "true").lower() == "true"

app = FastAPI(
    title="Neural Search API",
    description=(
        "Unified neural search endpoint over Linked Data knowledge graphs. "
        "Provides semantic entity discovery for AI agents and RAG pipelines."
    ),
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    """Neural search query."""

    query: str = Field(..., description="Natural language search query")
    limit: int = Field(10, ge=1, le=100, description="Max results to return")
    page: int = Field(0, ge=0, description="Page number for pagination")
    entity_type: str = Field("", description="Optional Schema.org type filter (full URI or short name)")
    backend: str = Field(
        "graphql",
        description="Search backend: 'graphql' (direct entitySearch) or 'mcp' (WordLift MCP)",
    )


class EntityResult(BaseModel):
    """A single entity result."""

    uri: str = Field(..., description="Entity IRI (dereferenceable)")
    name: str = Field("", description="Entity name")
    url: str = Field("", description="Canonical web page URL")
    types: list[str] = Field(default_factory=list, description="RDF types")
    description: str = Field("", description="Short description")
    score: float = Field(0.0, description="Relevance score")
    source: str = Field("", description="Source backend")


class SearchResponse(BaseModel):
    """Search response."""

    query: str
    results: list[EntityResult]
    total: int
    backend: str
    latency_ms: float
    timestamp: str


# ---------------------------------------------------------------------------
# Search backends
# ---------------------------------------------------------------------------


async def search_via_mcp(
    query: str,
    api_key: str,
    limit: int = 10,
    entity_type: str = "",
) -> list[EntityResult]:
    """Search via the WordLift MCP Neural Search tool."""
    payload = {
        "method": "tools/call",
        "params": {
            "name": "neural-search",
            "arguments": {
                "query": query,
                "limit": limit,
            },
        },
    }
    if entity_type:
        payload["params"]["arguments"]["entity_type"] = entity_type

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{WORDLIFT_MCP_ENDPOINT}/messages",
            json=payload,
            headers={
                "Authorization": f"Key {api_key}",
                "Content-Type": "application/json",
            },
        )
        resp.raise_for_status()
        data = resp.json()

    # Parse MCP response into EntityResult objects
    results = []
    if isinstance(data, dict):
        content = data.get("result", {}).get("content", [])
        for item in content:
            if item.get("type") == "text":
                try:
                    entity_data = json.loads(item["text"])
                    results.append(EntityResult(
                        uri=entity_data.get("@id", entity_data.get("iri", "")),
                        name=entity_data.get("name", ""),
                        url=entity_data.get("url", ""),
                        types=entity_data.get("@type", []) if isinstance(
                            entity_data.get("@type"), list
                        ) else [entity_data.get("@type", "")],
                        description=entity_data.get("description", "")[:200],
                        score=entity_data.get("score", 0.0),
                        source="mcp",
                    ))
                except (json.JSONDecodeError, TypeError):
                    results.append(EntityResult(
                        uri="",
                        name=item["text"][:100],
                        description=item["text"][:200],
                        source="mcp",
                    ))
    return results


def _build_entity_search_query(
    query: str,
    limit: int = 10,
    page: int = 0,
    entity_type: str = "",
) -> str:
    """Build the GraphQL entitySearch query using the WordLift schema.

    Uses the real entitySearch endpoint with proper field selectors:
      - iri: entity IRI
      - refs(name: "rdf:type"): RDF types
      - string(name: "schema:url"): canonical URL
      - string(name: "schema:description"): description
      - float(name: "_:score"): relevance score
    """
    # Escape the query string for GraphQL
    escaped = query.replace("\\", "\\\\").replace('"', '\\"')

    # Build the type constraint clause if specified
    type_constraint = ""
    if entity_type:
        # Accept both short names and full URIs
        if not entity_type.startswith("http"):
            entity_type = f"http://schema.org/{entity_type}"
        type_constraint = f'typeConstraint: {{ in: ["{entity_type}"] }}'

    # Build the query block
    query_block_parts = [f'search: {{ string: "{escaped}" }}']
    if type_constraint:
        query_block_parts.append(type_constraint)
    query_block = ", ".join(query_block_parts)

    return f"""{{
  entitySearch(
    page: {page}
    rows: {limit}
    query: {{ {query_block} }}
  ) {{
    iri
    types: refs(name: "rdf:type")
    url: string(name: "schema:url")
    name: string(name: "schema:name")
    description: string(name: "schema:description")
    score: float(name: "_:score")
  }}
}}"""


async def search_via_graphql(
    query: str,
    api_key: str,
    limit: int = 10,
    page: int = 0,
    entity_type: str = "",
) -> list[EntityResult]:
    """Search via direct GraphQL entitySearch against the WordLift KG."""
    graphql_query = _build_entity_search_query(query, limit, page, entity_type)

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            GRAPHQL_ENDPOINT,
            json={"query": graphql_query},
            headers={
                "Authorization": f"Key {api_key}",
                "Content-Type": "application/json",
            },
        )
        resp.raise_for_status()
        data = resp.json()

    # Check for GraphQL errors
    if "errors" in data:
        error_msgs = [e.get("message", "") for e in data["errors"]]
        logger.warning("GraphQL errors: %s", error_msgs)

    results = []
    entities = data.get("data", {}).get("entitySearch", []) or []
    for entity in entities:
        results.append(EntityResult(
            uri=entity.get("iri", ""),
            name=entity.get("name", ""),
            url=entity.get("url", ""),
            types=entity.get("types", []) or [],
            description=(entity.get("description") or "")[:200],
            score=entity.get("score") or 0.0,
            source="graphql",
        ))
    return results


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/search", response_model=SearchResponse)
async def neural_search(
    req: SearchRequest,
    authorization: str = Header(..., description="API key: 'Key sk-...'"),
):
    """Perform a neural search across a knowledge graph.

    Accepts a natural language query and returns ranked entities
    with URIs, types, and relevance scores. Each entity URI is
    dereferenceable via content negotiation (append .json, .ttl, .html).

    Authentication: Pass `Key <your-api-key>` in the Authorization header.
    """
    api_key = authorization.removeprefix("Key ").removeprefix("Bearer ").strip()
    if not api_key:
        raise HTTPException(status_code=401, detail="API key required")

    start = time.monotonic()

    try:
        if req.backend == "graphql":
            results = await search_via_graphql(
                req.query, api_key, req.limit, req.page, req.entity_type
            )
        else:
            results = await search_via_mcp(
                req.query, api_key, req.limit, req.entity_type
            )
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"Backend error: {exc.response.text[:200]}",
        )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Backend unreachable: {exc}")

    latency_ms = (time.monotonic() - start) * 1000

    if LOG_QUERIES:
        logger.info(
            "query=%r backend=%s results=%d latency=%.1fms",
            req.query, req.backend, len(results), latency_ms,
        )

    return SearchResponse(
        query=req.query,
        results=results,
        total=len(results),
        backend=req.backend,
        latency_ms=round(latency_ms, 1),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/health")
async def health():
    """Health check for Cloud Run."""
    return {"status": "ok", "service": "neural-search-api"}


@app.get("/")
async def root():
    """Service info and usage instructions for AI agents."""
    return {
        "service": "Neural Search API",
        "description": (
            "Semantic entity search over Linked Data knowledge graphs. "
            "Each result URI supports content negotiation — append .json, .ttl, or .html."
        ),
        "endpoints": {
            "POST /search": "Neural search with query, limit, entity_type, backend params",
            "GET /health": "Health check",
        },
        "authentication": "Authorization: Key <your-api-key>",
        "example": {
            "query": "sustainable tourism activities in Salzburg",
            "limit": 5,
            "entity_type": "TouristAttraction",
            "backend": "mcp",
        },
    }
