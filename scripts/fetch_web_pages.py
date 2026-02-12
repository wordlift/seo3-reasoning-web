"""Fetch the corresponding web pages for all collected entities.

Reads entity.json files, extracts schema:url, and fetches the actual HTML
web page. Saves as page.html alongside the entity files.
"""
import json
import glob
import logging
import sys
from pathlib import Path

import httpx

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def extract_page_url(entity_json_path: str) -> str | None:
    """Extract the web page URL from an entity's JSON-LD."""
    with open(entity_json_path) as f:
        data = json.load(f)

    entities = data if isinstance(data, list) else [data]
    for entity in entities:
        # Try schema:url first, then mainEntityOfPage
        for prop in ["http://schema.org/url", "http://schema.org/mainEntityOfPage"]:
            val = entity.get(prop, [])
            if isinstance(val, list) and val:
                # Could be @id or @value
                first = val[0]
                url = first.get("@id") or first.get("@value", "")
                if url and url.startswith("http"):
                    return url
            elif isinstance(val, str) and val.startswith("http"):
                return val
    return None


def main():
    domains = sys.argv[1:] if len(sys.argv) > 1 else ["wordlift_blog", "blackbriar", "salzburgerland"]
    raw_dir = Path("data/raw")

    client = httpx.Client(timeout=30.0, follow_redirects=True)
    total_fetched = 0
    total_skipped = 0
    total_failed = 0

    try:
        for domain in domains:
            domain_dir = raw_dir / domain
            if not domain_dir.exists():
                logger.warning("Domain dir not found: %s", domain_dir)
                continue

            entity_files = sorted(glob.glob(str(domain_dir / "*/entity.json")))
            logger.info("=== %s: %d entities ===", domain, len(entity_files))

            seen_urls: set[str] = set()
            for entity_file in entity_files:
                entity_dir = Path(entity_file).parent
                page_path = entity_dir / "page.html"

                # Skip if already fetched
                if page_path.exists():
                    total_skipped += 1
                    continue

                url = extract_page_url(entity_file)
                if not url:
                    logger.warning("  No URL found in %s", entity_file)
                    total_skipped += 1
                    continue

                # Deduplicate URLs (product variants share the same page)
                base_url = url.split("?")[0]
                if base_url in seen_urls:
                    # Save a reference instead
                    ref_path = entity_dir / "page_url.txt"
                    ref_path.write_text(url)
                    total_skipped += 1
                    continue
                seen_urls.add(base_url)

                try:
                    resp = client.get(url)
                    resp.raise_for_status()
                    page_path.write_bytes(resp.content)
                    total_fetched += 1
                    logger.info("  Saved page.html ← %s (%d bytes)", base_url, len(resp.content))
                except httpx.HTTPError as exc:
                    logger.warning("  Failed: %s — %s", base_url, exc)
                    # Save the URL for reference
                    ref_path = entity_dir / "page_url.txt"
                    ref_path.write_text(url)
                    total_failed += 1

    finally:
        client.close()

    logger.info("=== Summary ===")
    logger.info("  Fetched: %d pages", total_fetched)
    logger.info("  Skipped: %d (already fetched or duplicate)", total_skipped)
    logger.info("  Failed:  %d", total_failed)


if __name__ == "__main__":
    main()
