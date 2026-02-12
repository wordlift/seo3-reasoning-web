"""Batch collector — downloads entity HTML + JSON-LD from the LDP and
runs C6-Plus transformation for all domains.

Works WITHOUT API keys by dereferencing entity IRIs directly from the
WordLift Linked Data Platform.

Usage:
    python scripts/collect_and_transform.py          # all domains
    python scripts/collect_and_transform.py wordlift_blog  # single domain
"""

import json
import logging
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset.transformer import EntityTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


# ---------------------------------------------------------------------------
# LDP entity IRIs — known entities per domain
# ---------------------------------------------------------------------------
# These are the entities we know exist in each domain's KG.
# For a full production run, you'd discover these via GraphQL.
DOMAIN_ENTITIES = {
    "wordlift_blog": {
        "account_id": "wl0216",
        "domain_name": "WordLift Blog",
        "base_url": "https://data.wordlift.io/wl0216",
        "entities": [
            {"id": "entity/knowledge-graph", "type": "Thing"},
            {"id": "entity/andrea_volpini", "type": "Person"},
            {"id": "entity/structured-data", "type": "Thing"},
            {"id": "entity/google", "type": "Organization"},
            {"id": "entity/schema-org", "type": "Thing"},
            {"id": "entity/natural-language-processing", "type": "Thing"},
            {"id": "entity/linked-data", "type": "Thing"},
            {"id": "entity/search-engine-optimization", "type": "Thing"},
            {"id": "entity/artificial-intelligence", "type": "Thing"},
            {"id": "entity/semantic-web", "type": "Thing"},
        ],
    },
    "express_legal_funding": {
        "account_id": "wl156383",
        "domain_name": "Express Legal Funding",
        "base_url": "https://data.wordlift.io/wl156383",
        "entities": [
            {"id": "entity/pre-settlement-funding", "type": "FinancialProduct"},
            {"id": "entity/lawsuit-loans", "type": "FinancialProduct"},
            {"id": "entity/express-legal-funding", "type": "Organization"},
            {"id": "entity/personal-injury", "type": "Thing"},
            {"id": "entity/car-accident", "type": "Thing"},
        ],
    },
    "salzburgerland": {
        "account_id": "salzburgerland",
        "domain_name": "SalzburgerLand",
        "base_url": "http://open.salzburgerland.com/de",
        "entities": [
            {"id": "LodgingBusiness/pritzhuette-1-800-m", "type": "LodgingBusiness"},
            {"id": "LodgingBusiness/hotel-goldener-hirsch", "type": "LodgingBusiness"},
            {"id": "TouristAttraction/festung-hohensalzburg", "type": "TouristAttraction"},
            {"id": "Place/salzburg", "type": "Place"},
            {"id": "Place/zell-am-see", "type": "Place"},
        ],
    },
    "blackbriar": {
        "account_id": "wl172055",
        "domain_name": "BlackBriar",
        "base_url": "https://data.wordlift.io/wl172055",
        "entities": [
            {"id": "entity/blackbriar", "type": "Brand"},
            {"id": "entity/backpack", "type": "Product"},
            {"id": "entity/duffel-bag", "type": "Product"},
        ],
    },
}


def fetch_entity(
    client: httpx.Client,
    base_url: str,
    entity_id: str,
    output_dir: Path,
    formats: dict | None = None,
) -> dict:
    """Fetch an entity from the LDP in multiple formats.

    Args:
        client: HTTP client.
        base_url: LDP base URL (e.g. https://data.wordlift.io/wl0216).
        entity_id: Entity ID (e.g. entity/knowledge-graph).
        output_dir: Where to save fetched files.
        formats: Dict of format_name -> (accept_header, file_ext).

    Returns:
        Dict with fetch results.
    """
    if formats is None:
        formats = {
            "html": ("text/html", ".html"),
            "jsonld": ("application/ld+json", ".json"),
            "turtle": ("text/turtle", ".ttl"),
            "rdfxml": ("application/rdf+xml", ".rdf"),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    iri = f"{base_url.rstrip('/')}/{entity_id}"
    result = {"entity_id": entity_id, "iri": iri, "formats": {}}

    for fmt_name, (accept, ext) in formats.items():
        url = f"{iri}{ext}" if ext != ".html" else iri
        filepath = output_dir / f"entity{ext}"

        try:
            resp = client.get(
                url,
                headers={"Accept": accept},
                follow_redirects=True,
                timeout=30,
            )
            if resp.status_code == 200 and len(resp.content) > 100:
                filepath.write_bytes(resp.content)
                result["formats"][fmt_name] = {
                    "path": str(filepath),
                    "size": len(resp.content),
                    "status": resp.status_code,
                }
                logger.info("  ✓ %s: %d bytes", fmt_name, len(resp.content))
            else:
                logger.warning("  ✗ %s: HTTP %d (%d bytes)", fmt_name, resp.status_code, len(resp.content))
                result["formats"][fmt_name] = {
                    "status": resp.status_code,
                    "error": f"HTTP {resp.status_code}",
                }
        except httpx.HTTPError as e:
            logger.warning("  ✗ %s: %s", fmt_name, e)
            result["formats"][fmt_name] = {"error": str(e)}

        time.sleep(0.3)  # Be polite

    # Also try requesting without extension (content negotiation)
    for fmt_name, (accept, ext) in formats.items():
        filepath = output_dir / f"entity{ext}"
        if filepath.exists():
            continue  # Already got it

        try:
            resp = client.get(
                iri,
                headers={"Accept": accept},
                follow_redirects=True,
                timeout=30,
            )
            if resp.status_code == 200 and len(resp.content) > 100:
                filepath.write_bytes(resp.content)
                result["formats"][fmt_name] = {
                    "path": str(filepath),
                    "size": len(resp.content),
                    "status": resp.status_code,
                    "via": "content_negotiation",
                }
                logger.info("  ✓ %s (via conneg): %d bytes", fmt_name, len(resp.content))
            time.sleep(0.3)
        except httpx.HTTPError:
            pass

    return result


def collect_domain(domain_key: str, domain_cfg: dict) -> list[dict]:
    """Collect all entities for a domain."""
    logger.info("=" * 60)
    logger.info("Collecting domain: %s", domain_cfg["domain_name"])
    logger.info("=" * 60)

    domain_dir = RAW_DIR / domain_key
    results = []

    with httpx.Client() as client:
        for entity in domain_cfg["entities"]:
            entity_slug = entity["id"].replace("/", "_")
            entity_dir = domain_dir / entity_slug
            logger.info("Entity: %s", entity["id"])

            result = fetch_entity(
                client=client,
                base_url=domain_cfg["base_url"],
                entity_id=entity["id"],
                output_dir=entity_dir,
            )
            result["type"] = entity.get("type", "Thing")
            results.append(result)

    return results


def transform_domain(domain_key: str, domain_name: str):
    """Run C6-Plus transformation for all entities in a domain."""
    logger.info("Transforming domain: %s", domain_name)
    transformer = EntityTransformer(
        raw_dir=RAW_DIR,
        output_dir=PROCESSED_DIR,
        templates_dir=PROJECT_ROOT / "templates",
    )
    results = transformer.process_domain(domain_key)
    logger.info("  Processed %d entities", len(results))

    for r in results:
        logger.info("  Entity: %s", r["entity"])
        for k, v in r.items():
            if k != "entity":
                logger.info("    %s: %s", k, v)

    return results


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else None

    domains = DOMAIN_ENTITIES
    if target:
        if target in domains:
            domains = {target: domains[target]}
        else:
            logger.error("Unknown domain: %s. Available: %s", target, list(DOMAIN_ENTITIES.keys()))
            sys.exit(1)

    # Phase 1: Collect
    all_results = {}
    for key, cfg in domains.items():
        results = collect_domain(key, cfg)
        all_results[key] = results

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Collection Summary")
    logger.info("=" * 60)
    for domain, results in all_results.items():
        successful = sum(1 for r in results if any(
            f.get("path") for f in r.get("formats", {}).values()
        ))
        logger.info("  %s: %d/%d entities collected", domain, successful, len(results))

    # Phase 2: Transform
    logger.info("\n" + "=" * 60)
    logger.info("Transformation Phase")
    logger.info("=" * 60)
    for key in domains:
        transform_domain(key, domains[key]["domain_name"])


if __name__ == "__main__":
    main()
