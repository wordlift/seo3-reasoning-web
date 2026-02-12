"""Collector module — fetches entity pages from the Linked Data Platform.

Uses content negotiation to retrieve entities in multiple formats
(HTML, JSON-LD, RDF/XML, Turtle) and stores them in data/raw/.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import httpx
import yaml

from config.settings import (
    ACCEPT_HTML,
    ACCEPT_JSONLD,
    ACCEPT_RDFXML,
    ACCEPT_TURTLE,
    LDP_DATA_API,
    LDP_ENTITY_BASE,
    RAW_DIR,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Content-negotiation format registry
# ---------------------------------------------------------------------------
FORMATS = {
    "html": {"accept": ACCEPT_HTML, "ext": ".html"},
    "jsonld": {"accept": ACCEPT_JSONLD, "ext": ".json"},
    "turtle": {"accept": ACCEPT_TURTLE, "ext": ".ttl"},
    "rdfxml": {"accept": ACCEPT_RDFXML, "ext": ".rdf"},
}


class EntityCollector:
    """Fetches entity data from a Linked Data Platform via content negotiation."""

    def __init__(
        self,
        output_dir: Path = RAW_DIR,
        timeout: float = 30.0,
    ) -> None:
        self.output_dir = output_dir
        self.client = httpx.Client(timeout=timeout, follow_redirects=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect_entity_by_page_url(
        self,
        page_url: str,
        domain_name: str,
    ) -> dict:
        """Fetch structured data for a page URL via the data API.

        Args:
            page_url: The canonical webpage URL (e.g. https://www.zurichna.com/insurance/accident/student)
            domain_name: Human-readable domain label used for organizing output.

        Returns:
            Dict with paths to the saved files.
        """
        # Build the data-API URL:  api.wordlift.io/data/https/www.example.com/path
        stripped = page_url.removeprefix("https://").removeprefix("http://")
        api_url = f"{LDP_DATA_API}/https/{stripped}"

        logger.info("Fetching structured data for %s via %s", page_url, api_url)

        result = {"page_url": page_url, "api_url": api_url, "files": {}}
        slug = stripped.replace("/", "_").replace(".", "_")
        entity_dir = self.output_dir / domain_name / slug
        entity_dir.mkdir(parents=True, exist_ok=True)

        # Fetch JSON-LD from the data API
        try:
            resp = self.client.get(api_url)
            resp.raise_for_status()
            data = resp.json()
            out_path = entity_dir / "structured_data.json"
            out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
            result["files"]["jsonld_api"] = str(out_path)
            logger.info("  Saved JSON-LD API data → %s", out_path)
        except httpx.HTTPError as exc:
            logger.warning("  Failed to fetch data API: %s", exc)

        return result

    def collect_entity_by_id(
        self,
        account_id: str,
        entity_path: str,
        domain_name: str,
        entity_base_url: str = "",
    ) -> dict:
        """Fetch an entity page from the Linked Data Platform in all formats.

        Args:
            account_id: The LDP account identifier (e.g. "wl0216").
            entity_path: Relative entity path (e.g. "entity/knowledge-graph").
            domain_name: Human-readable domain label.
            entity_base_url: Custom entity base URL (e.g. for SalzburgerLand).

        Returns:
            Dict with paths to the saved files per format.
        """
        if entity_base_url:
            base_url = f"{entity_base_url.rstrip('/')}/{entity_path}"
        else:
            base_url = f"{LDP_ENTITY_BASE}/{account_id}/{entity_path}"
        logger.info("Fetching entity %s", base_url)

        result = {"entity_url": base_url, "files": {}}
        slug = entity_path.replace("/", "_")
        entity_dir = self.output_dir / domain_name / slug
        entity_dir.mkdir(parents=True, exist_ok=True)

        for fmt_name, fmt_cfg in FORMATS.items():
            url = base_url + fmt_cfg["ext"] if fmt_name != "html" else base_url + ".html"
            try:
                resp = self.client.get(url)
                resp.raise_for_status()
                out_path = entity_dir / f"entity{fmt_cfg['ext']}"
                out_path.write_bytes(resp.content)
                result["files"][fmt_name] = str(out_path)
                logger.info("  Saved %s → %s", fmt_name, out_path)
            except httpx.HTTPError as exc:
                logger.warning("  Failed to fetch %s format: %s", fmt_name, exc)

        return result

    def collect_entity_by_iri(
        self,
        iri: str,
        domain_name: str,
    ) -> dict:
        """Fetch an entity by its full IRI in all formats.

        This is the primary method for entities discovered via GraphQL entitySearch,
        which returns full IRIs rather than account_id + path pairs.

        Args:
            iri: Full entity IRI (e.g. "https://data.wordlift.io/wl0216/entity/knowledge-graph").
            domain_name: Human-readable domain label.

        Returns:
            Dict with paths to the saved files per format.
        """
        logger.info("Fetching entity by IRI: %s", iri)

        result = {"entity_url": iri, "files": {}}
        # Create slug from IRI path
        from urllib.parse import urlparse
        parsed = urlparse(iri)
        slug = parsed.path.strip("/").replace("/", "_")
        entity_dir = self.output_dir / domain_name / slug
        entity_dir.mkdir(parents=True, exist_ok=True)

        for fmt_name, fmt_cfg in FORMATS.items():
            url = iri + fmt_cfg["ext"] if fmt_name != "html" else iri + ".html"
            try:
                resp = self.client.get(url)
                resp.raise_for_status()
                out_path = entity_dir / f"entity{fmt_cfg['ext']}"
                out_path.write_bytes(resp.content)
                result["files"][fmt_name] = str(out_path)
                logger.info("  Saved %s → %s", fmt_name, out_path)
            except httpx.HTTPError as exc:
                logger.warning("  Failed to fetch %s format: %s", fmt_name, exc)

        return result

    def discover_via_graphql(
        self,
        api_key: str,
        queries: list[str],
        entity_types: list[str] | None = None,
        limit_per_query: int = 10,
    ) -> list[dict]:
        """Discover entities using the WordLift GraphQL entitySearch API.

        Runs multiple search queries to build a diverse entity set.

        Args:
            api_key: WordLift API key.
            queries: List of search queries to run.
            entity_types: Optional Schema.org type filters.
            limit_per_query: Max results per query.

        Returns:
            List of entity dicts with iri, name, types, score.
        """
        seen_iris: set[str] = set()
        entities: list[dict] = []

        for query in queries:
            # Build type constraint
            type_constraint = ""
            if entity_types:
                type_list = ", ".join(f'"{t}"' for t in entity_types)
                type_constraint = f"typeConstraint: {{ in: [{type_list}] }}"

            # Escape query
            escaped = query.replace("\\", "\\\\").replace('"', '\\"')

            # Build query block
            query_parts = [f'search: {{ string: "{escaped}" }}']
            if type_constraint:
                query_parts.append(type_constraint)
            query_block = ", ".join(query_parts)

            graphql_query = f"""{{
  entitySearch(
    page: 0
    rows: {limit_per_query}
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

            try:
                resp = self.client.post(
                    "https://api.wordlift.io/graphql",
                    json={"query": graphql_query},
                    headers={
                        "Authorization": f"Key {api_key}",
                        "Content-Type": "application/json",
                    },
                )
                resp.raise_for_status()
                data = resp.json()

                for entity in data.get("data", {}).get("entitySearch", []) or []:
                    iri = entity.get("iri", "")
                    if iri and iri not in seen_iris:
                        seen_iris.add(iri)
                        entities.append(entity)
                        logger.info("  Discovered: %s (%.3f)", entity.get("name", iri), entity.get("score", 0))

            except httpx.HTTPError as exc:
                logger.warning("  GraphQL search failed for %r: %s", query, exc)

        logger.info("  Total unique entities discovered: %d", len(entities))
        return entities

    def collect_entities_from_page_html(
        self,
        page_url: str,
        domain_name: str,
        account_id: str = "",
    ) -> list[dict]:
        """Discover entities by scraping JSON-LD from a live webpage.

        This follows the real-world entity discovery flow:
          1. Fetch the live HTML page
          2. Extract all <script type="application/ld+json"> blocks
          3. Find @id values matching data.wordlift.io/{account_id}/...
          4. Dereference each entity ID via content negotiation

        Args:
            page_url: URL of the webpage to scrape.
            domain_name: Human-readable domain label.
            account_id: Expected LDP account ID for filtering entity URIs.

        Returns:
            List of collection result dicts.
        """
        import re

        from bs4 import BeautifulSoup

        logger.info("Discovering entities from page HTML: %s", page_url)

        # Step 1: Fetch the page
        try:
            resp = self.client.get(page_url)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.error("  Failed to fetch page: %s", exc)
            return []

        # Step 2: Extract JSON-LD blocks
        soup = BeautifulSoup(resp.text, "lxml")
        jsonld_scripts = soup.find_all("script", {"type": "application/ld+json"})
        all_entities = []
        for script in jsonld_scripts:
            try:
                data = json.loads(script.string)
                if isinstance(data, list):
                    all_entities.extend(data)
                else:
                    all_entities.append(data)
            except (json.JSONDecodeError, TypeError):
                continue

        # Step 3: Find entity @id URIs on data.wordlift.io
        entity_ids = set()
        id_pattern = re.compile(r"https?://data\.wordlift\.io/([^/]+)/(.*)")
        for entity in all_entities:
            eid = entity.get("@id", "")
            m = id_pattern.match(eid)
            if m:
                found_account = m.group(1)
                entity_path = m.group(2)
                # Filter by account_id if specified
                if account_id and found_account != account_id:
                    continue
                entity_ids.add((found_account, entity_path))

        if not entity_ids:
            logger.warning("  No entity @id URIs found in page JSON-LD")
            # Fall back to data API
            return [self.collect_entity_by_page_url(page_url, domain_name)]

        logger.info("  Found %d entity IDs in JSON-LD", len(entity_ids))

        # Step 4: Dereference each entity via content negotiation
        results = []
        for acct, path in entity_ids:
            # Strip trailing extensions from path
            clean_path = path.removesuffix(".html").removesuffix(".json")
            r = self.collect_entity_by_id(acct, clean_path, domain_name)
            results.append(r)

        return results

    def collect_domain(self, domain_cfg: dict) -> list[dict]:
        """Collect all entities for a domain.

        Strategy:
          1. If API key is available, discover entities via GraphQL entitySearch.
          2. Dereference each discovered entity IRI in all formats.
          3. Also collect any pre-defined sample_entities.

        Args:
            domain_cfg: Domain configuration dictionary from domains.json.

        Returns:
            List of collection results.
        """
        from config.settings import get_wordlift_key

        domain_name = domain_cfg["name"].lower().replace(" ", "_")
        account_id = domain_cfg.get("account_id", "")
        entity_base_url = domain_cfg.get("entity_base_url", "")
        results = []

        # --- Phase 1: Discover entities via GraphQL ---
        api_key = get_wordlift_key(account_id) if account_id else ""
        if api_key:
            # Build diverse search queries per vertical
            vertical = domain_cfg.get("vertical", "")
            discovery_queries = self._get_discovery_queries(
                vertical, domain_cfg.get("entity_types", [])
            )

            logger.info("Discovering entities via GraphQL for %s (%d queries)",
                        domain_cfg["name"], len(discovery_queries))

            discovered = self.discover_via_graphql(
                api_key=api_key,
                queries=discovery_queries,
                limit_per_query=10,
            )

            # Dereference each discovered entity
            for entity in discovered:
                iri = entity.get("iri", "")
                if not iri:
                    continue
                r = self.collect_entity_by_iri(iri, domain_name)
                r["discovery"] = entity  # Store discovery metadata
                results.append(r)

        # --- Phase 2: Collect pre-defined sample entities ---
        for entity in domain_cfg.get("sample_entities", []):
            if domain_cfg.get("account_id"):
                r = self.collect_entity_by_id(
                    account_id=account_id,
                    entity_path=entity["entity_id"],
                    domain_name=domain_name,
                    entity_base_url=entity_base_url,
                )
                results.append(r)

            if entity.get("path"):
                page_url = domain_cfg["base_url"].rstrip("/") + entity["path"]
                r = self.collect_entity_by_page_url(page_url, domain_name)
                results.append(r)

        return results

    @staticmethod
    def _get_discovery_queries(vertical: str, entity_types: list[str]) -> list[str]:
        """Generate diverse search queries based on the domain vertical."""
        vertical_queries = {
            "editorial": [
                "knowledge graph",
                "structured data SEO",
                "entity optimization",
                "linked data web",
                "content marketing AI",
            ],
            "legal": [
                "pre-settlement funding",
                "lawsuit loan",
                "legal finance",
                "personal injury settlement",
                "litigation funding",
            ],
            "travel": [
                "hotel accommodation",
                "tourist attraction",
                "restaurant dining",
                "hiking outdoor activity",
                "ski resort winter sport",
            ],
            "ecommerce": [
                "backpack outdoor",
                "travel bag luggage",
                "expedition gear",
                "duffel bag",
                "accessories clothing",
            ],
        }
        queries = vertical_queries.get(vertical, ["product", "service", "organization"])
        # Add entity type names as queries too
        for et in entity_types[:3]:
            queries.append(et.lower())
        return queries

    def close(self) -> None:
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Collect entity data from Linked Data Platform")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/experiment_config.yaml"),
        help="Path to experiment config YAML",
    )
    parser.add_argument("--domain", type=str, help="Collect only this domain (by name)")
    parser.add_argument("--limit", type=int, help="Max entities to collect per domain")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print what would be collected without fetching"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load domain details
    domains_file = Path("data/domains.json")
    with open(domains_file) as f:
        domains_data = json.load(f)

    domains = domains_data["domains"]
    if args.domain:
        domains = [d for d in domains if d["name"].lower() == args.domain.lower()]

    if args.dry_run:
        for d in domains:
            n = len(d.get("sample_entities", []))
            print(f"  {d['name']}: {n} entities")
        print("Dry run — no data fetched.")
        return

    with EntityCollector() as collector:
        for domain in domains:
            logger.info("=== Collecting: %s ===", domain["name"])
            results = collector.collect_domain(domain)
            logger.info("  Collected %d items", len(results))

            # Save manifest
            manifest_dir = RAW_DIR / domain["name"].lower().replace(" ", "_")
            manifest_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = manifest_dir / "manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(results, f, indent=2)
            logger.info("  Manifest → %s", manifest_path)


if __name__ == "__main__":
    main()
