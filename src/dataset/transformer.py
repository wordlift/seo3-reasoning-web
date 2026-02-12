"""Transformer module — creates the 4 document variants per entity.

Variant 1 (Plain HTML):      Raw HTML stripped of JSON-LD script blocks.
Variant 2 (HTML + JSON-LD):  Original HTML with embedded structured data.
Variant 3 (Enhanced):        Agentic-optimized HTML with natural-language summary,
                              embedded JSON-LD, visible linked entity navigation,
                              llms.txt agent instructions, and neural search SKILL.
Variant 4 (Enhanced Plus):   Original WordLift HTML augmented with summary block
                              and llms.txt agent instructions (C6-Plus).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml
from bs4 import BeautifulSoup
from jinja2 import Environment, FileSystemLoader

from config.settings import PROCESSED_DIR, RAW_DIR, TEMPLATES_DIR

logger = logging.getLogger(__name__)


class EntityTransformer:
    """Creates document variants from raw entity data."""

    def __init__(
        self,
        raw_dir: Path = RAW_DIR,
        output_dir: Path = PROCESSED_DIR,
        templates_dir: Path = TEMPLATES_DIR,
    ) -> None:
        self.raw_dir = raw_dir
        self.output_dir = output_dir
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(templates_dir)),
            autoescape=True,
        )

    # ------------------------------------------------------------------
    # Variant generators
    # ------------------------------------------------------------------

    def create_plain_html(self, html_content: str) -> str:
        """Variant 1: Strip all JSON-LD script blocks from HTML."""
        soup = BeautifulSoup(html_content, "lxml")

        # Remove all <script type="application/ld+json"> blocks
        for script in soup.find_all("script", {"type": "application/ld+json"}):
            script.decompose()

        return str(soup)

    def create_html_with_jsonld(self, html_content: str, jsonld_data: list | dict) -> str:
        """Variant 2: Ensure HTML has embedded JSON-LD structured data.

        If the HTML already contains JSON-LD, return as-is.
        Otherwise, inject the provided JSON-LD data into a <script> block.
        """
        soup = BeautifulSoup(html_content, "lxml")
        existing = soup.find_all("script", {"type": "application/ld+json"})

        if existing:
            return str(soup)

        # Inject JSON-LD
        script_tag = soup.new_tag("script", type="application/ld+json")
        script_tag.string = json.dumps(jsonld_data, ensure_ascii=False)

        head = soup.find("head")
        if head:
            head.append(script_tag)
        else:
            soup.append(script_tag)

        return str(soup)

    def create_enhanced_entity_page(
        self,
        jsonld_data: list | dict,
        entity_url: str,
        domain_name: str,
    ) -> str:
        """Variant 3: Agentic-optimized entity page.

        Combines:
        - Natural language summary from structured data
        - Embedded JSON-LD block
        - Visible linked entity navigation
        - llms.txt agent instructions
        - Neural search SKILL reference
        - Breadcrumb hierarchy
        """
        # Flatten JSON-LD (handle @graph wrappers, nested lists)
        entities = self._flatten_jsonld(jsonld_data)

        # Find the main entity (the one with schema:name or the first typed entity)
        main_entity = self._find_main_entity(entities)
        if not main_entity:
            logger.warning("No main entity found in JSON-LD data for %s", entity_url)
            main_entity = entities[0] if entities else {}

        # Extract properties
        entity_name = self._get_value(main_entity, "name", "schema:name", "http://schema.org/name")
        entity_desc = self._get_value(
            main_entity, "description", "schema:description", "http://schema.org/description"
        )
        entity_types = self._get_types(main_entity)
        linked_entities = self._extract_links(entities)
        offers = self._extract_offers(entities)
        faq_items = self._extract_faq(entities)

        # Build breadcrumb from types
        breadcrumb = self._build_breadcrumb(entity_types)

        # Generate llms.txt instructions
        llms_instructions = self._generate_llms_instructions(
            entity_name=entity_name,
            entity_url=entity_url,
            linked_entities=linked_entities,
            domain_name=domain_name,
        )

        # Render template
        template = self.jinja_env.get_template("enhanced_entity.html")
        return template.render(
            entity_name=entity_name or "Unknown Entity",
            entity_description=entity_desc or "",
            entity_types=entity_types,
            entity_url=entity_url,
            breadcrumb=breadcrumb,
            jsonld_data=json.dumps(entities, indent=2, ensure_ascii=False),
            linked_entities=linked_entities,
            offers=offers,
            faq_items=faq_items,
            llms_instructions=llms_instructions,
            domain_name=domain_name,
        )

    def create_enhanced_plus_page(
        self,
        html_content: str,
        jsonld_data: list | dict,
        entity_url: str,
        domain_name: str,
    ) -> str:
        """Variant 4 (C6-Plus): Generate a Wikidata-style structured entity page.

        Produces a fully structured page with:
        1. Entity header (name, types, canonical + entity URIs)
        2. Description section
        3. Statements table (public properties extracted from LDP HTML)
        4. Related Entities (derived from statement links)
        5. Sitelinks (canonical web page + data format links)
        6. Agent instructions with search endpoint reference
        """
        # Flatten JSON-LD
        entities = self._flatten_jsonld(jsonld_data)

        main_entity = self._find_main_entity(entities)
        if not main_entity:
            main_entity = entities[0] if entities else {}

        entity_name = self._get_value(main_entity, "name", "schema:name", "http://schema.org/name")
        entity_desc = self._get_value(
            main_entity, "description", "schema:description", "http://schema.org/description"
        )
        entity_types = self._get_types(main_entity)

        # Extract canonical URL
        canonical_url = self._extract_canonical_url(entities, entity_url)

        # Extract statements from LDP HTML table (filtered)
        statements = self._extract_statements_from_html(html_content)

        # Derive linked entities from statements (all rows with entity URLs)
        # This is much more comprehensive than _extract_links which only checks
        # a hardcoded list of JSON-LD properties
        linked_entities = []
        seen_urls = set()
        # Build name map from JSON-LD graph
        entity_names_map = {}
        for ent in entities:
            eid = ent.get("@id", "")
            ename = self._get_value(ent, "name", "schema:name", "http://schema.org/name")
            if eid and ename:
                entity_names_map[eid] = ename

        for stmt in statements:
            url = stmt.get("value_url", "")
            prop = stmt.get("prop_label", "")
            if not url or url in seen_urls:
                continue
            # Only include links to entity-like URIs (not schema.org types, etc.)
            if not url.startswith("http"):
                continue
            if "schema.org/" in url and not "data.wordlift.io" in url and not "open." in url:
                continue
            seen_urls.add(url)
            # Use name from graph, or derive from URL
            name = entity_names_map.get(url, "")
            if not name:
                name = url.rsplit("/", 1)[-1].replace("_", " ").replace("-", " ").title()
            html_url = url + ".html" if not url.endswith(".html") else url
            linked_entities.append({
                "relation": prop,
                "url": url,
                "html_url": html_url,
                "name": name,
            })

        # Determine primary type for search instructions
        primary_type = entity_types[0] if entity_types else "Thing"

        # Generate llms.txt instructions via template
        try:
            llms_template = self.jinja_env.get_template("llms_instructions.md")
            llms_instructions = llms_template.render(
                entity_name=entity_name or "Unknown Entity",
                entity_url=entity_url,
                linked_entities=linked_entities,
                domain_name=domain_name,
                primary_type=primary_type,
            )
        except Exception:
            llms_instructions = self._generate_llms_instructions(
                entity_name=entity_name,
                entity_url=entity_url,
                linked_entities=linked_entities,
                domain_name=domain_name,
            )

        # Render JSON-LD block for embedding in <head>
        import json as _json
        jsonld_block = _json.dumps(jsonld_data, indent=2, ensure_ascii=False) if jsonld_data else ""

        # Render full structured page via template
        template = self.jinja_env.get_template("enhanced_entity_plus.html")
        return template.render(
            entity_name=entity_name or "Unknown Entity",
            entity_description=entity_desc or "",
            entity_types=entity_types,
            entity_url=entity_url,
            canonical_url=canonical_url,
            statements=statements,
            linked_entities=linked_entities,
            llms_instructions=llms_instructions,
            jsonld_block=jsonld_block,
            domain_name=domain_name,
            primary_type=primary_type,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    # Prefixes that indicate private/internal KG properties
    PRIVATE_PROPERTY_PREFIXES = (
        "wordpress:",
        "http://purl.org/wordpress/",
        "seovoc:",
        "http://schema.org/seovoc",
        "https://schema.org/seovoc",
    )

    def _extract_canonical_url(self, entities: list[dict], entity_url: str) -> str:
        """Extract the canonical web page URL for this entity.

        Priority:
        1. schema:url / http://schema.org/url — the main web page
        2. wordpress:permalink from the RDFa (via JSON-LD wp props)
        3. Fall back to entity_url (@id) if it looks like a web page
        """
        for entity in entities:
            # Check schema:url variants
            url = self._get_value(
                entity,
                "url", "schema:url", "http://schema.org/url",
            )
            if url and url.startswith("http"):
                return url

            # Check wordpress:permalink
            permalink = entity.get(
                "wordpress:permalink",
                entity.get("http://purl.org/wordpress/1.0/permalink", ""),
            )
            if permalink:
                # Handle list values (some backends return arrays)
                if isinstance(permalink, list):
                    permalink = permalink[0] if permalink else ""
                if isinstance(permalink, dict):
                    permalink = permalink.get("@value", permalink.get("@id", ""))
                permalink = str(permalink)
                # Strip xsd:string type annotation if present
                if "^^" in permalink:
                    permalink = permalink.split("^^")[0].strip('"')
                if permalink.startswith("http"):
                    return permalink

        # Fallback: if entity_url looks like a web page (not a data URI)
        if entity_url and "data.wordlift.io" not in entity_url:
            return entity_url

        return ""

    def _strip_private_properties(self, soup: BeautifulSoup) -> None:
        """Remove elements with private RDFa properties from the HTML.

        Handles three patterns:
        1. RDFa: elements with property="wordpress:..." or rel="seovoc:..."
        2. LDP tables: <tr> rows where the property cell text starts with
           "wordpress:" or "seovoc:" (e.g. <td><a>wordpress:content</a></td>)
        3. Links: <a> tags whose text or href contains private property URIs
        """
        # URI prefixes for private properties (used in href attributes)
        private_uri_prefixes = (
            "http://purl.org/wordpress/",
            "https://vocab.summaryofme.com/",  # seovoc namespace
            "http://schema.org/seovoc",
            "https://schema.org/seovoc",
        )

        # 1. Remove elements with private RDFa property attributes
        for tag in soup.find_all(attrs={"property": True}):
            prop = tag.get("property", "")
            if any(prop.startswith(prefix) for prefix in self.PRIVATE_PROPERTY_PREFIXES):
                tag.decompose()

        # 2. Remove elements with private RDFa `rel` attributes
        for tag in soup.find_all(attrs={"rel": True}):
            rel = tag.get("rel", "")
            if isinstance(rel, list):
                rel = " ".join(rel)
            if any(rel.startswith(prefix) for prefix in self.PRIVATE_PROPERTY_PREFIXES):
                tag.decompose()

        # 3. Remove table rows where property text starts with private prefix
        #    LDP renders as: <tr><td><a href="...">wordpress:content</a></td><td>...</td></tr>
        for tr in soup.find_all("tr"):
            first_td = tr.find("td")
            if first_td:
                cell_text = first_td.get_text(strip=True)
                if any(cell_text.startswith(prefix) for prefix in self.PRIVATE_PROPERTY_PREFIXES):
                    tr.decompose()

        # 4. Remove standalone links whose text or href points to private properties
        for a in list(soup.find_all("a")):
            text = (a.string or "").strip()
            href = a.get("href", "")
            if (
                any(text.startswith(prefix) for prefix in self.PRIVATE_PROPERTY_PREFIXES)
                or any(href.startswith(prefix) for prefix in private_uri_prefixes)
            ):
                # If inside a <td>, remove the whole row
                parent_tr = a.find_parent("tr")
                if parent_tr:
                    parent_tr.decompose()
                else:
                    a.decompose()

    def _extract_statements_from_html(self, html_content: str) -> list[dict]:
        """Extract structured property–value pairs from the LDP entity HTML.

        Parses the property table in the LDP HTML (rendered by data.wordlift.io),
        filters out private properties (wordpress:*, seovoc:*), and returns a list
        of statement dicts for the Wikidata-style template:

            [{"prop_label": "schema:name", "prop_url": "...", "value_label": "...", "value_url": "..."}, ...]
        """
        soup = BeautifulSoup(html_content, "lxml")
        statements = []

        for tr in soup.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) < 2:
                continue

            # First <td> = property
            prop_td = tds[0]
            prop_link = prop_td.find("a")
            prop_label = prop_td.get_text(strip=True)
            prop_url = prop_link["href"] if prop_link and prop_link.get("href") else ""

            # Skip private properties
            if any(prop_label.startswith(prefix) for prefix in self.PRIVATE_PROPERTY_PREFIXES):
                continue
            if prop_url and any(prop_url.startswith(prefix) for prefix in (
                "http://purl.org/wordpress/",
                "https://vocab.summaryofme.com/",
            )):
                continue

            # Second <td> = value
            value_td = tds[1]
            value_link = value_td.find("a")
            value_label = value_td.get_text(strip=True)
            value_url = value_link["href"] if value_link and value_link.get("href") else ""

            if not prop_label or not value_label:
                continue

            statements.append({
                "prop_label": prop_label,
                "prop_url": prop_url,
                "value_label": value_label,
                "value_url": value_url,
            })

        return statements

    @staticmethod
    def _flatten_jsonld(data: list | dict) -> list[dict]:
        """Flatten JSON-LD data into a list of entities.

        Handles:
        - @graph wrappers (WordPress Yoast/SEO plugins)
        - Nested lists (multiple JSON-LD blocks extracted from page)
        - Single dict entities
        """
        result = []
        items = data if isinstance(data, list) else [data]
        for item in items:
            if isinstance(item, list):
                result.extend(EntityTransformer._flatten_jsonld(item))
            elif isinstance(item, dict):
                if "@graph" in item:
                    graph = item["@graph"]
                    if isinstance(graph, list):
                        result.extend(graph)
                    elif isinstance(graph, dict):
                        result.append(graph)
                else:
                    result.append(item)
        return result

    def _find_main_entity(self, entities: list[dict]) -> dict | None:
        """Find the primary entity in a JSON-LD graph (the one with a name)."""
        for e in entities:
            if any(
                k in e
                for k in ["name", "schema:name", "http://schema.org/name"]
            ):
                types = self._get_types(e)
                # Prefer non-WebPage, non-Answer entities
                if not any(t in ("WebPage", "Answer", "WebSite", "FAQPage") for t in types):
                    return e
        # Fallback: first entity with a name
        for e in entities:
            if any(k in e for k in ["name", "schema:name", "http://schema.org/name"]):
                return e
        return None

    @staticmethod
    def _get_value(entity: dict, *keys: str) -> str | None:
        """Get a scalar value from an entity, trying multiple key variations."""
        for key in keys:
            val = entity.get(key)
            if val is None:
                continue
            if isinstance(val, list):
                # JSON-LD expanded form: [{"@value": "..."}]
                for item in val:
                    if isinstance(item, dict) and "@value" in item:
                        return item["@value"]
                    if isinstance(item, str):
                        return item
            if isinstance(val, dict) and "@value" in val:
                return val["@value"]
            if isinstance(val, str):
                return val
        return None

    @staticmethod
    def _get_types(entity: dict) -> list[str]:
        """Extract Schema.org type names from an entity."""
        raw = entity.get("@type", entity.get("rdf:type", []))
        if isinstance(raw, str):
            raw = [raw]
        types = []
        for t in raw:
            if isinstance(t, str):
                # "http://schema.org/Product" → "Product"
                name = t.rsplit("/", 1)[-1] if "/" in t else t
                types.append(name)
        return types

    @staticmethod
    def _extract_links(entities: list[dict]) -> list[dict]:
        """Extract linked entity references from the JSON-LD graph."""
        links = []
        seen_ids = set()
        link_properties = [
            "provider", "schema:provider", "http://schema.org/provider",
            "offers", "schema:offers", "http://schema.org/offers",
            "sameAs", "schema:sameAs", "http://schema.org/sameAs",
            "about", "schema:about", "http://schema.org/about",
            "isPartOf", "schema:isPartOf", "http://schema.org/isPartOf",
            "publisher", "schema:publisher", "http://schema.org/publisher",
            "mainEntityOfPage", "schema:mainEntityOfPage", "http://schema.org/mainEntityOfPage",
        ]
        for entity in entities:
            for prop in link_properties:
                val = entity.get(prop)
                if val is None:
                    continue
                refs = val if isinstance(val, list) else [val]
                for ref in refs:
                    if isinstance(ref, dict) and "@id" in ref:
                        entity_id = ref["@id"]
                        if entity_id not in seen_ids:
                            seen_ids.add(entity_id)
                            label = prop.rsplit("/", 1)[-1] if "/" in prop else prop
                            links.append({
                                "url": entity_id,
                                "relation": label,
                                "html_url": entity_id + ".html"
                                if not entity_id.endswith(".html")
                                else entity_id,
                            })
        return links

    @staticmethod
    def _extract_offers(entities: list[dict]) -> list[dict]:
        """Extract offer/pricing information from the entity graph."""
        offers = []
        for e in entities:
            types = e.get("@type", [])
            if isinstance(types, str):
                types = [types]
            type_names = [t.rsplit("/", 1)[-1] for t in types]
            if any(t in ("Offer", "PriceSpecification") for t in type_names):
                offers.append({
                    "min_price": e.get("minPrice", e.get("http://schema.org/minPrice")),
                    "max_price": e.get("maxPrice", e.get("http://schema.org/maxPrice")),
                    "currency": e.get(
                        "priceCurrency", e.get("http://schema.org/priceCurrency")
                    ),
                    "description": e.get(
                        "description", e.get("http://schema.org/description")
                    ),
                    "availability": e.get(
                        "availability", e.get("http://schema.org/availability")
                    ),
                })
        return offers

    @staticmethod
    def _extract_faq(entities: list[dict]) -> list[dict]:
        """Extract FAQ question-answer pairs from the entity graph."""
        questions = {}
        answers = {}
        for e in entities:
            types = e.get("@type", [])
            if isinstance(types, str):
                types = [types]
            type_names = [t.rsplit("/", 1)[-1] for t in types]
            eid = e.get("@id", "")
            if "Question" in type_names:
                text = e.get("name", e.get("text", e.get("http://schema.org/name", "")))
                if isinstance(text, list):
                    text = text[0] if text else ""
                questions[eid] = text
                # Link to answer
                ans_ref = e.get(
                    "acceptedAnswer", e.get("http://schema.org/acceptedAnswer", {})
                )
                if isinstance(ans_ref, dict):
                    questions[eid] = {"text": text, "answer_id": ans_ref.get("@id")}
            elif "Answer" in type_names:
                text = e.get("text", e.get("http://schema.org/text", ""))
                answers[eid] = text

        faq = []
        for qid, qdata in questions.items():
            if isinstance(qdata, dict):
                ans_text = answers.get(qdata.get("answer_id", ""), "")
                faq.append({"question": qdata["text"], "answer": ans_text})
            else:
                faq.append({"question": qdata, "answer": ""})
        return faq

    @staticmethod
    def _build_breadcrumb(types: list[str]) -> list[str]:
        """Build a type hierarchy breadcrumb."""
        hierarchy = ["Thing"]
        for t in types:
            if t != "Thing" and t not in hierarchy:
                hierarchy.append(t)
        return hierarchy

    def _generate_llms_instructions(
        self,
        entity_name: str | None,
        entity_url: str,
        linked_entities: list[dict],
        domain_name: str,
    ) -> str:
        """Generate llms.txt-style instructions for an agent exploring this entity."""
        lines = [
            f"# Agent Instructions for {entity_name or 'this entity'}",
            "",
            f"> This page describes **{entity_name}** as part of the {domain_name} knowledge graph.",
            "",
            "## How to use this page",
            "",
            "1. The JSON-LD block below contains machine-readable structured data about this entity.",
            "2. Use the linked entities section to explore related concepts, providers, and offers.",
            "3. Each linked entity URL supports content negotiation — append `.json` for JSON-LD, "
            "`.ttl` for Turtle, or `.html` for a human-readable view.",
            "",
        ]

        if linked_entities:
            lines.append("## Related entities to explore")
            lines.append("")
            for le in linked_entities:
                lines.append(f"- **{le['relation']}**: [{le['url']}]({le['html_url']})")
            lines.append("")

        lines.extend([
            "## Neural Search",
            "",
            "To find related entities in this knowledge graph, you can query the neural search",
            "endpoint with a natural language question. This allows you to discover entities",
            "that are semantically related but not directly linked.",
            "",
        ])

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def process_entity(self, entity_dir: Path, domain_name: str) -> dict:
        """Process a single entity directory, creating all 3 variants.

        Source files:
          - page.html: The actual web page from the live site (for variants 1-2)
          - entity.html: LDP entity HTML page (fallback for variants 1-2)
          - entity.json: LDP JSON-LD data (for variant 3 enhanced, and for injecting into variant 2)

        Args:
            entity_dir: Path to data/raw/{domain}/{entity_slug}/
            domain_name: Domain label.

        Returns:
            Dict mapping variant name → output path.
        """
        result = {}
        out_dir = self.output_dir / domain_name / entity_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Load raw data — prefer actual web page for HTML variants
        page_path = entity_dir / "page.html"
        entity_html_path = entity_dir / "entity.html"
        jsonld_path = entity_dir / "entity.json"
        api_jsonld_path = entity_dir / "structured_data.json"

        # HTML content: prefer real web page, fall back to LDP entity page
        html_content = ""
        if page_path.exists():
            html_content = page_path.read_text(errors="replace")
        elif entity_html_path.exists():
            html_content = entity_html_path.read_text(errors="replace")

        # JSON-LD data: prefer LDP entity JSON, fall back to API data
        jsonld_data = []
        if jsonld_path.exists():
            try:
                jsonld_data = json.loads(jsonld_path.read_text())
            except json.JSONDecodeError:
                pass
        if not jsonld_data and api_jsonld_path.exists():
            try:
                jsonld_data = json.loads(api_jsonld_path.read_text())
            except json.JSONDecodeError:
                pass

        entity_url = ""
        if jsonld_data:
            first = jsonld_data[0] if isinstance(jsonld_data, list) else jsonld_data
            entity_url = first.get("@id", "")

        # Variant 1: Plain HTML (web page stripped of JSON-LD)
        if html_content:
            plain = self.create_plain_html(html_content)
            p = out_dir / "plain.html"
            p.write_text(plain)
            result["plain_html"] = str(p)

        # Variant 2: HTML + JSON-LD (web page with structured data)
        if html_content and jsonld_data:
            with_ld = self.create_html_with_jsonld(html_content, jsonld_data)
            p = out_dir / "with_jsonld.html"
            p.write_text(with_ld)
            result["html_jsonld"] = str(p)

        # Variant 3: Enhanced entity page (agentic-optimized from JSON-LD)
        if jsonld_data:
            enhanced = self.create_enhanced_entity_page(
                jsonld_data=jsonld_data,
                entity_url=entity_url,
                domain_name=domain_name,
            )
            p = out_dir / "enhanced.html"
            p.write_text(enhanced)
            result["enhanced"] = str(p)

        # Variant 4: C6-Plus — KG entity HTML + summary + agent instructions
        # Priority for base HTML: entity.html (LDP) > enhanced (C6) > page.html
        # This ensures C6-Plus works for any website with a WordLift KG,
        # even when the LDP entity HTML is not available.
        if jsonld_data:
            # Choose the best base HTML for C6-Plus
            c6plus_base_html = ""
            if entity_html_path.exists():
                # Best: use the LDP entity HTML (domain-agnostic, always available)
                c6plus_base_html = entity_html_path.read_text(errors="replace")
                logger.info("  C6-Plus base: entity.html (LDP)")
            elif "enhanced" in result:
                # Fallback: use the C6 enhanced page we just generated
                c6plus_base_html = Path(result["enhanced"]).read_text(errors="replace")
                logger.info("  C6-Plus base: enhanced.html (C6 generated)")
            elif html_content:
                # Last resort: use the website page HTML
                c6plus_base_html = html_content
                logger.info("  C6-Plus base: page.html (website)")

            if c6plus_base_html:
                enhanced_plus = self.create_enhanced_plus_page(
                    html_content=c6plus_base_html,
                    jsonld_data=jsonld_data,
                    entity_url=entity_url,
                    domain_name=domain_name,
                )
                p = out_dir / "enhanced_plus.html"
                p.write_text(enhanced_plus)
                result["enhanced_plus"] = str(p)

        return result

    def process_domain(self, domain_name: str) -> list[dict]:
        """Process all entities for a domain."""
        domain_dir = self.raw_dir / domain_name
        if not domain_dir.exists():
            logger.warning("Raw data directory not found: %s", domain_dir)
            return []

        results = []
        for entity_dir in sorted(domain_dir.iterdir()):
            if entity_dir.is_dir() and entity_dir.name != "manifest.json":
                logger.info("Processing %s / %s", domain_name, entity_dir.name)
                r = self.process_entity(entity_dir, domain_name)
                results.append({"entity": entity_dir.name, **r})

        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Create document variants from raw entity data")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/experiment_config.yaml"),
    )
    parser.add_argument("--domain", type=str, help="Process only this domain")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    transformer = EntityTransformer()

    for domain_cfg in config["domains"]:
        name = domain_cfg["name"].lower().replace(" ", "_")
        if args.domain and name != args.domain.lower().replace(" ", "_"):
            continue
        logger.info("=== Transforming: %s ===", domain_cfg["name"])
        results = transformer.process_domain(name)
        logger.info("  Transformed %d entities", len(results))


if __name__ == "__main__":
    main()
