"""Factual Overlap Analysis: C1 (Plain HTML) vs C3 (Enhanced Entity Page).

Quantifies the information overlap between the baseline and enhanced document
representations to address the reviewer's concern about confounded treatment
effects (information content vs. presentation format).

For each entity, extracts "facts" from both C1 and C3 documents and computes:
  - Facts in C1 only
  - Facts in C3 only (novel KG-derived information)
  - Facts in both (shared)
  - Jaccard similarity
  - Presentation-only elements in C3 (breadcrumbs, agent instructions, etc.)

Usage:
    python scripts/factual_overlap.py [--output results/factual_overlap.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import PROCESSED_DIR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fact extraction
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, collapse whitespace, strip XSD annotations."""
    if not text:
        return ""
    # Strip XSD type annotations like "value"^^xsd:string
    text = re.sub(r'"\^\^xsd:\w+', '', text)
    text = re.sub(r'\^\^xsd:\w+', '', text)
    # Strip surrounding quotes
    text = text.strip('"').strip("'")
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text


def extract_facts_from_plain_html(html_content: str) -> dict:
    """Extract structured facts from C1 (plain HTML / LDP entity page with JSON-LD stripped).

    The LDP entity page renders properties as an HTML table:
        <tr><td><a href="schema:name">schema:name</a></td><td>value</td></tr>

    Returns a dict with:
      - 'properties': set of (property_name, value) tuples
      - 'entity_links': set of linked entity URLs
      - 'text_content': set of normalized text segments
    """
    soup = BeautifulSoup(html_content, "lxml")
    properties = set()
    entity_links = set()
    text_content = set()

    # Extract from property table rows
    for tr in soup.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 2:
            continue

        prop_td = tds[0]
        value_td = tds[1]

        # Property name
        prop_text = prop_td.get_text(strip=True)
        # Skip private/internal properties
        if any(prop_text.startswith(prefix) for prefix in (
            "wordpress:", "http://purl.org/wordpress/", "seovoc:",
        )):
            continue

        # Value: text content
        value_text = normalize_text(value_td.get_text(strip=True))

        # Value: linked URL
        value_link = value_td.find("a")
        value_url = value_link["href"] if value_link and value_link.get("href") else ""

        if value_text and value_text not in ('""', ""):
            properties.add((prop_text, value_text))
            # Add individual words/phrases to text_content for overlap comparison
            if len(value_text) > 3:
                text_content.add(value_text)

        if value_url and value_url.startswith("http"):
            entity_links.add(value_url)

    # Also extract any visible text outside the table
    for p in soup.find_all(["p", "h1", "h2", "h3", "span"]):
        text = normalize_text(p.get_text(strip=True))
        if text and len(text) > 5:
            text_content.add(text)

    return {
        "properties": properties,
        "entity_links": entity_links,
        "text_content": text_content,
    }


def extract_facts_from_enhanced_html(html_content: str) -> dict:
    """Extract structured facts from C3 (enhanced entity page).

    The enhanced page has structured sections:
      - Description section
      - Offers section
      - FAQ section
      - Linked Entities section
      - Agent Instructions section (presentation-only)
      - Breadcrumbs (presentation-only)
      - JSON-LD block (embedded data)

    Returns:
      - 'properties': set of (property_name, value) tuples
      - 'entity_links': set of linked entity URLs
      - 'text_content': set of normalized text segments
      - 'presentation_elements': set of presentation-only elements
      - 'jsonld_properties': set of properties from the JSON-LD block
    """
    soup = BeautifulSoup(html_content, "lxml")
    properties = set()
    entity_links = set()
    text_content = set()
    presentation_elements = set()
    jsonld_properties = set()

    # 1. Entity name
    h1 = soup.find("h1")
    if h1:
        name = normalize_text(h1.get_text(strip=True))
        if name:
            properties.add(("name", name))
            text_content.add(name)

    # 2. Description
    desc_el = soup.find("p", class_="description")
    if desc_el:
        desc = normalize_text(desc_el.get_text(strip=True))
        if desc:
            properties.add(("description", desc))
            text_content.add(desc)

    # 3. Entity type
    type_div = soup.find("div", class_="entity-type")
    if type_div:
        type_text = normalize_text(type_div.get_text(strip=True))
        if type_text:
            properties.add(("type", type_text))

    # 4. Breadcrumbs (presentation-only)
    breadcrumb = soup.find("nav", class_="breadcrumb")
    if breadcrumb:
        presentation_elements.add("breadcrumb_navigation")

    # 5. Linked entities
    linked_section = soup.find("section", class_="linked-entities")
    if linked_section:
        for li in linked_section.find_all("li"):
            rel_span = li.find("span", class_="relation")
            link = li.find("a")
            if rel_span and link:
                relation = normalize_text(rel_span.get_text(strip=True).rstrip(":"))
                url = link.get("href", "")
                if url:
                    entity_links.add(url)
                    properties.add(("linked_entity_" + relation, url))

    # 6. Offers
    for offer in soup.find_all("div", class_="offer-card"):
        offer_text = normalize_text(offer.get_text(strip=True))
        if offer_text:
            properties.add(("offer", offer_text))
            text_content.add(offer_text)

    # 7. FAQ
    for faq in soup.find_all("div", class_="faq-item"):
        q_el = faq.find("h3")
        a_el = faq.find("p")
        if q_el:
            q = normalize_text(q_el.get_text(strip=True))
            properties.add(("faq_question", q))
            text_content.add(q)
        if a_el:
            a = normalize_text(a_el.get_text(strip=True))
            properties.add(("faq_answer", a))
            text_content.add(a)

    # 8. Agent instructions (presentation-only)
    agent_section = soup.find("section", class_="agent-instructions")
    if agent_section:
        presentation_elements.add("agent_instructions")

    # 9. JSON-LD embedded block
    for script in soup.find_all("script", {"type": "application/ld+json"}):
        try:
            raw_text = script.string or ""
            # Unescape HTML entities
            import html
            raw_text = html.unescape(raw_text)
            data = json.loads(raw_text)
            items = data if isinstance(data, list) else [data]
            for item in items:
                if isinstance(item, dict):
                    for key, val in item.items():
                        if key.startswith("@"):
                            continue
                        prop_name = key.rsplit("/", 1)[-1] if "/" in key else key
                        if isinstance(val, list):
                            for v in val:
                                if isinstance(v, dict):
                                    vtext = v.get("@value", v.get("@id", ""))
                                else:
                                    vtext = str(v)
                                if vtext:
                                    jsonld_properties.add((prop_name, normalize_text(str(vtext))))
                        elif isinstance(val, dict):
                            vtext = val.get("@value", val.get("@id", ""))
                            if vtext:
                                jsonld_properties.add((prop_name, normalize_text(str(vtext))))
                        elif isinstance(val, str):
                            jsonld_properties.add((prop_name, normalize_text(val)))
        except (json.JSONDecodeError, Exception):
            pass

    return {
        "properties": properties,
        "entity_links": entity_links,
        "text_content": text_content,
        "presentation_elements": presentation_elements,
        "jsonld_properties": jsonld_properties,
    }


# ---------------------------------------------------------------------------
# Overlap computation
# ---------------------------------------------------------------------------

def compute_text_overlap(c1_texts: set, c3_texts: set) -> dict:
    """Compute text-level overlap between C1 and C3.

    Uses substring matching: a C1 text is considered present in C3
    if any C3 text contains it (or vice versa).
    """
    c1_in_c3 = set()
    c3_in_c1 = set()

    for c1t in c1_texts:
        for c3t in c3_texts:
            if c1t in c3t or c3t in c1t:
                c1_in_c3.add(c1t)
                c3_in_c1.add(c3t)
                break

    for c3t in c3_texts:
        for c1t in c1_texts:
            if c3t in c1t or c1t in c3t:
                c3_in_c1.add(c3t)
                break

    shared = c1_in_c3  # C1 items found in C3
    c1_only = c1_texts - c1_in_c3
    c3_only = c3_texts - c3_in_c1

    union = len(c1_texts | c3_texts) if (c1_texts | c3_texts) else 1
    jaccard = len(shared) / union if union else 0.0

    return {
        "shared_count": len(shared),
        "c1_only_count": len(c1_only),
        "c3_only_count": len(c3_only),
        "c1_total": len(c1_texts),
        "c3_total": len(c3_texts),
        "jaccard": jaccard,
        "c1_coverage": len(c1_in_c3) / len(c1_texts) if c1_texts else 1.0,
        "c3_coverage": len(c3_in_c1) / len(c3_texts) if c3_texts else 1.0,
        "c1_only_items": sorted(c1_only)[:5],  # Sample for reporting
        "c3_only_items": sorted(c3_only)[:5],
    }


def compute_link_overlap(c1_links: set, c3_links: set) -> dict:
    """Compute entity link overlap between C1 and C3."""
    # Normalize links: strip trailing .html for comparison
    def norm_link(url):
        return url.rstrip("/").removesuffix(".html")

    c1_norm = {norm_link(u) for u in c1_links}
    c3_norm = {norm_link(u) for u in c3_links}

    shared = c1_norm & c3_norm
    c1_only = c1_norm - c3_norm
    c3_only = c3_norm - c1_norm
    union = len(c1_norm | c3_norm) if (c1_norm | c3_norm) else 1

    return {
        "shared_count": len(shared),
        "c1_only_count": len(c1_only),
        "c3_only_count": len(c3_only),
        "jaccard": len(shared) / union if union else 0.0,
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_entity(entity_dir: Path) -> dict | None:
    """Analyze factual overlap for a single entity."""
    plain_path = entity_dir / "plain.html"
    enhanced_path = entity_dir / "enhanced.html"

    if not plain_path.exists() or not enhanced_path.exists():
        return None

    c1_html = plain_path.read_text(errors="replace")
    c3_html = enhanced_path.read_text(errors="replace")

    c1_facts = extract_facts_from_plain_html(c1_html)
    c3_facts = extract_facts_from_enhanced_html(c3_html)

    text_overlap = compute_text_overlap(c1_facts["text_content"], c3_facts["text_content"])
    link_overlap = compute_link_overlap(c1_facts["entity_links"], c3_facts["entity_links"])

    return {
        "c1_properties": len(c1_facts["properties"]),
        "c3_properties": len(c3_facts["properties"]),
        "c1_text_segments": len(c1_facts["text_content"]),
        "c3_text_segments": len(c3_facts["text_content"]),
        "c1_entity_links": len(c1_facts["entity_links"]),
        "c3_entity_links": len(c3_facts["entity_links"]),
        "c3_presentation_elements": len(c3_facts["presentation_elements"]),
        "c3_presentation_list": ", ".join(sorted(c3_facts["presentation_elements"])),
        "c3_jsonld_properties": len(c3_facts["jsonld_properties"]),
        "text_shared": text_overlap["shared_count"],
        "text_c1_only": text_overlap["c1_only_count"],
        "text_c3_only": text_overlap["c3_only_count"],
        "text_jaccard": text_overlap["jaccard"],
        "text_c1_coverage": text_overlap["c1_coverage"],
        "text_c3_coverage": text_overlap["c3_coverage"],
        "link_shared": link_overlap["shared_count"],
        "link_c1_only": link_overlap["c1_only_count"],
        "link_c3_only": link_overlap["c3_only_count"],
        "link_jaccard": link_overlap["jaccard"],
        "text_c1_only_sample": "; ".join(text_overlap["c1_only_items"][:3]),
        "text_c3_only_sample": "; ".join(text_overlap["c3_only_items"][:3]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Factual overlap analysis between C1 and C3 documents")
    parser.add_argument("--output", type=Path, default=Path("results/factual_overlap.csv"))
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Collect results
    all_results = []
    domain_summaries = {}

    for domain_dir in sorted(args.processed_dir.iterdir()):
        if not domain_dir.is_dir():
            continue
        domain_name = domain_dir.name
        domain_results = []

        for entity_dir in sorted(domain_dir.iterdir()):
            if not entity_dir.is_dir():
                continue

            result = analyze_entity(entity_dir)
            if result is None:
                continue

            result["domain"] = domain_name
            result["entity"] = entity_dir.name
            all_results.append(result)
            domain_results.append(result)

        # Domain summary
        if domain_results:
            n = len(domain_results)
            avg_text_jaccard = sum(r["text_jaccard"] for r in domain_results) / n
            avg_c1_coverage = sum(r["text_c1_coverage"] for r in domain_results) / n
            avg_c3_coverage = sum(r["text_c3_coverage"] for r in domain_results) / n
            avg_link_jaccard = sum(r["link_jaccard"] for r in domain_results) / n
            total_c3_only_text = sum(r["text_c3_only"] for r in domain_results)
            total_c1_only_text = sum(r["text_c1_only"] for r in domain_results)
            avg_presentation = sum(r["c3_presentation_elements"] for r in domain_results) / n

            domain_summaries[domain_name] = {
                "entities": n,
                "avg_text_jaccard": avg_text_jaccard,
                "avg_c1_coverage": avg_c1_coverage,
                "avg_c3_coverage": avg_c3_coverage,
                "avg_link_jaccard": avg_link_jaccard,
                "total_c3_only_text": total_c3_only_text,
                "total_c1_only_text": total_c1_only_text,
                "avg_presentation_elements": avg_presentation,
            }

    # Write per-entity CSV
    if all_results:
        fieldnames = [
            "domain", "entity",
            "c1_properties", "c3_properties",
            "c1_text_segments", "c3_text_segments",
            "c1_entity_links", "c3_entity_links",
            "c3_presentation_elements", "c3_presentation_list",
            "c3_jsonld_properties",
            "text_shared", "text_c1_only", "text_c3_only",
            "text_jaccard", "text_c1_coverage", "text_c3_coverage",
            "link_shared", "link_c1_only", "link_c3_only", "link_jaccard",
            "text_c1_only_sample", "text_c3_only_sample",
        ]
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

        logger.info("Wrote per-entity results to %s (%d entities)", args.output, len(all_results))

    # Print domain summary
    print("\n" + "=" * 80)
    print("FACTUAL OVERLAP ANALYSIS: C1 (Plain HTML) vs C3 (Enhanced Entity Page)")
    print("=" * 80)

    for domain, summary in sorted(domain_summaries.items()):
        print(f"\n--- {domain} ({summary['entities']} entities) ---")
        print(f"  Text Jaccard similarity:     {summary['avg_text_jaccard']:.3f}")
        print(f"  C1 coverage in C3:           {summary['avg_c1_coverage']:.1%}")
        print(f"  C3 coverage in C1:           {summary['avg_c3_coverage']:.1%}")
        print(f"  Link Jaccard similarity:     {summary['avg_link_jaccard']:.3f}")
        print(f"  Total C1-only text segments: {summary['total_c1_only_text']}")
        print(f"  Total C3-only text segments: {summary['total_c3_only_text']}")
        print(f"  Avg presentation elements:   {summary['avg_presentation_elements']:.1f}")

    # Overall summary
    if all_results:
        n = len(all_results)
        overall_text_jaccard = sum(r["text_jaccard"] for r in all_results) / n
        overall_c1_cov = sum(r["text_c1_coverage"] for r in all_results) / n
        overall_c3_cov = sum(r["text_c3_coverage"] for r in all_results) / n
        overall_link_jaccard = sum(r["link_jaccard"] for r in all_results) / n

        print(f"\n{'=' * 80}")
        print(f"OVERALL ({n} entities)")
        print(f"  Average text Jaccard:  {overall_text_jaccard:.3f}")
        print(f"  C1 content in C3:      {overall_c1_cov:.1%} (how much of C1's facts appear in C3)")
        print(f"  C3 content in C1:      {overall_c3_cov:.1%} (how much of C3's facts appear in C1)")
        print(f"  Average link Jaccard:  {overall_link_jaccard:.3f}")
        print(f"{'=' * 80}")

        # Key finding for the paper
        print("\n📝 KEY FINDING FOR PAPER:")
        if overall_c1_cov > 0.8:
            print(f"  C1 coverage by C3 is {overall_c1_cov:.1%} — the vast majority of C1 facts")
            print("  are also present in C3. The information asymmetry is SMALL.")
            print("  The performance gap is primarily attributable to presentation format.")
        elif overall_c1_cov > 0.5:
            print(f"  C1 coverage by C3 is {overall_c1_cov:.1%} — moderate overlap.")
            print("  Both information content and presentation contribute to the gap.")
        else:
            print(f"  C1 coverage by C3 is {overall_c1_cov:.1%} — substantial information asymmetry.")
            print("  The confound is significant and the paper should be revised accordingly.")


if __name__ == "__main__":
    main()
