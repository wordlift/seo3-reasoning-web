"""Central configuration for the Structured LD Agentic RAG project."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # Load .env for API keys

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
QUERIES_DIR = DATA_DIR / "queries"
RESULTS_DIR = PROJECT_ROOT / "results"
TEMPLATES_DIR = PROJECT_ROOT / "templates"

# ---------------------------------------------------------------------------
# Google Cloud
# ---------------------------------------------------------------------------
GCP_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "gen-lang-client-0470307714")
GCP_PROJECT_NUMBER = os.getenv("GOOGLE_CLOUD_PROJECT_NUMBER", "676736758338")
GCP_REGION = os.getenv("GCP_REGION", "us-central1")

# ---------------------------------------------------------------------------
# Vertex AI Models
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "text-embedding-005"
GENERATION_MODEL = "gemini-2.5-flash"
JUDGE_MODEL = "gemini-3-flash-preview"  # Gemini 3 family — different from generator (2.5) to reduce bias

# ---------------------------------------------------------------------------
# Vector Search 2.0
# ---------------------------------------------------------------------------
VECTOR_SEARCH_LOCATION = GCP_REGION
TOP_K = 10  # Default number of results to retrieve

# Collection names per experimental condition
COLLECTION_PREFIX = "sld-agentic-rag"
CONDITIONS = {
    "C1": "plain_html_standard",
    "C2": "jsonld_standard",
    "C3": "enhanced_standard",
    "C4": "plain_html_agentic",
    "C5": "jsonld_agentic",
    "C6": "enhanced_agentic",
    "C6_PLUS": "enhanced_plus_agentic",
}

# ---------------------------------------------------------------------------
# Agentic RAG
# ---------------------------------------------------------------------------
MAX_TRAVERSAL_HOPS = 2
MAX_LINKS_PER_HOP = 5
AGENT_TIMEOUT_SECONDS = 120

# ---------------------------------------------------------------------------
# Linked Data Platform endpoints
# ---------------------------------------------------------------------------
LDP_DATA_API = "https://api.wordlift.io/data"
LDP_ENTITY_BASE = "https://data.wordlift.io"

# Known entity URL patterns (used for link extraction in agentic RAG)
# Domains may publish entities at custom URLs — add patterns here.
ENTITY_URL_PATTERNS: list[str] = [
    r"https?://data\.wordlift\.io/[^\s\"<>'\}\])]+",
    r"https?://open\.salzburgerland\.com/de/[^\s\"<>'\}\])]+",
]

# Content negotiation headers
ACCEPT_JSONLD = "application/ld+json"
ACCEPT_HTML = "text/html"
ACCEPT_TURTLE = "text/turtle"
ACCEPT_RDFXML = "application/rdf+xml"

# ---------------------------------------------------------------------------
# WordLift MCP / Neural Search
# ---------------------------------------------------------------------------
WORDLIFT_MCP_ENDPOINT = "https://mcp.wordlift.io"

# Our own Cloud Run endpoint (set after deployment, falls back to MCP)
NEURAL_SEARCH_ENDPOINT = os.getenv("NEURAL_SEARCH_ENDPOINT", "")

# Per-domain API keys loaded from .env (pattern: WORDLIFT_KEY_{ACCOUNT_ID})
# Example: WORDLIFT_KEY_WL0216=sk-...
DOMAIN_API_KEYS: dict[str, str] = {}


def get_wordlift_key(account_id: str) -> str:
    """Get the WordLift API key for a specific knowledge graph.

    Looks up WORDLIFT_KEY_{ACCOUNT_ID} (uppercased) from environment.
    Keys must be stored in .env and never committed to source control.
    """
    env_var = f"WORDLIFT_KEY_{account_id.upper()}"
    key = os.getenv(env_var, "")
    if key:
        DOMAIN_API_KEYS[account_id] = key
    return key


@dataclass
class DomainConfig:
    """Configuration for a single domain under study."""

    name: str
    base_url: str
    vertical: str
    account_id: str  # e.g. "wl0216"
    entity_types: list[str] = field(default_factory=list)
    description: str = ""
    entity_base_url: str = ""  # Custom entity base URL (e.g. open.salzburgerland.com/de/)

    @property
    def data_api_base(self) -> str:
        """Base URL for the Linked Data Platform data API."""
        return f"{LDP_DATA_API}/https/{self.base_url.removeprefix('https://').removeprefix('http://')}"

    @property
    def entity_base(self) -> str:
        """Base URL for entity pages.

        Returns the custom entity_base_url if set (e.g. for SalzburgerLand),
        otherwise falls back to the standard data.wordlift.io/{account_id}.
        """
        if self.entity_base_url:
            return self.entity_base_url.rstrip("/")
        return f"{LDP_ENTITY_BASE}/{self.account_id}"

    @property
    def api_key(self) -> str:
        """WordLift API key for this domain's knowledge graph."""
        return get_wordlift_key(self.account_id)


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    domains: list[DomainConfig] = field(default_factory=list)
    top_k: int = TOP_K
    max_hops: int = MAX_TRAVERSAL_HOPS
    generation_model: str = GENERATION_MODEL
    judge_model: str = JUDGE_MODEL
    queries_per_entity: int = 3  # factual, relational, comparative
