"""Query generator — creates test queries with ground-truth answers.

For each entity, generates 3 query types:
  - Factual:     Direct attribute lookup from structured data
  - Relational:  Requires following links to related entities
  - Comparative: Requires reasoning across multiple entities
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from config.settings import GENERATION_MODEL, QUERIES_DIR, RAW_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates for LLM-assisted query generation
# ---------------------------------------------------------------------------

FACTUAL_PROMPT = """\
You are generating test queries for a retrieval-augmented generation (RAG) evaluation.

Given the following structured data (JSON-LD) about an entity, generate a **factual** question
that can be answered directly from the structured data fields (name, description, price, etc.).

Also provide the ground-truth answer based ONLY on the structured data.

JSON-LD:
{jsonld}

Respond in JSON format:
{{
  "query": "...",
  "ground_truth": "...",
  "required_fields": ["field1", "field2"]
}}
"""

RELATIONAL_PROMPT = """\
You are generating test queries for a RAG evaluation.

Given the following structured data about an entity and its linked entities,
generate a **relational** question that requires following at least one link
to a related entity (e.g., provider, offers, category) to answer fully.

Main entity JSON-LD:
{jsonld}

Linked entities (URLs):
{links}

Respond in JSON format:
{{
  "query": "...",
  "ground_truth": "...",
  "required_links": ["url1", "url2"],
  "reasoning": "Why this requires link traversal"
}}
"""

COMPARATIVE_PROMPT = """\
You are generating test queries for a RAG evaluation.

Given structured data about two entities from the same domain, generate a
**comparative** question that requires reasoning across both entities.

Entity A:
{jsonld_a}

Entity B:
{jsonld_b}

Respond in JSON format:
{{
  "query": "...",
  "ground_truth": "...",
  "entities_needed": ["entity_a_name", "entity_b_name"],
  "comparison_dimensions": ["dim1", "dim2"]
}}
"""


class QueryGenerator:
    """Generates test queries from structured entity data."""

    def __init__(
        self,
        raw_dir: Path = RAW_DIR,
        output_dir: Path = QUERIES_DIR,
        model: str = GENERATION_MODEL,
        use_llm: bool = True,
    ) -> None:
        self.raw_dir = raw_dir
        self.output_dir = output_dir
        self.model = model
        self.use_llm = use_llm
        self._genai_client = None

    @property
    def genai_client(self):
        if self._genai_client is None:
            from google import genai

            self._genai_client = genai.Client(
                vertexai=True,
                project="gen-lang-client-0470307714",
                location="us-central1",
            )
        return self._genai_client

    # ------------------------------------------------------------------
    # Template-based query generation (no LLM required)
    # ------------------------------------------------------------------

    def generate_factual_template(self, entity_data: dict) -> dict:
        """Generate a factual query using templates (no LLM)."""
        name = self._extract_name(entity_data)
        desc = self._extract_description(entity_data)

        if name:
            return {
                "type": "factual",
                "query": f"What is {name}? Describe its key features.",
                "ground_truth": desc or f"Information about {name}.",
                "source": "template",
            }
        return {}

    def generate_relational_template(self, entity_data: dict, links: list[dict]) -> dict:
        """Generate a relational query using templates (no LLM)."""
        name = self._extract_name(entity_data)
        if not name or not links:
            return {}

        # Pick a link with a meaningful relation
        for link in links:
            rel = link.get("relation", "")
            if rel in ("provider", "publisher", "offers"):
                return {
                    "type": "relational",
                    "query": f"Who is the {rel} of {name} and what else do they offer?",
                    "ground_truth": f"Requires following the {rel} link to {link['url']}.",
                    "required_links": [link["url"]],
                    "source": "template",
                }

        return {
            "type": "relational",
            "query": f"What entities are related to {name}?",
            "ground_truth": f"Related entities: {', '.join(l['url'] for l in links[:3])}.",
            "required_links": [l["url"] for l in links[:3]],
            "source": "template",
        }

    # ------------------------------------------------------------------
    # LLM-assisted query generation
    # ------------------------------------------------------------------

    def generate_factual_llm(self, jsonld_str: str) -> dict:
        """Generate a factual query using an LLM."""
        prompt = FACTUAL_PROMPT.format(jsonld=jsonld_str)
        response = self.genai_client.models.generate_content(
            model=self.model, contents=prompt
        )
        try:
            result = json.loads(response.text.strip().removeprefix("```json").removesuffix("```"))
            result["type"] = "factual"
            result["source"] = "llm"
            return result
        except (json.JSONDecodeError, AttributeError) as exc:
            logger.warning("Failed to parse LLM response for factual query: %s", exc)
            return {}

    def generate_relational_llm(self, jsonld_str: str, links: list[dict]) -> dict:
        """Generate a relational query using an LLM."""
        links_str = "\n".join(f"- {l['relation']}: {l['url']}" for l in links)
        prompt = RELATIONAL_PROMPT.format(jsonld=jsonld_str, links=links_str)
        response = self.genai_client.models.generate_content(
            model=self.model, contents=prompt
        )
        try:
            result = json.loads(response.text.strip().removeprefix("```json").removesuffix("```"))
            result["type"] = "relational"
            result["source"] = "llm"
            return result
        except (json.JSONDecodeError, AttributeError) as exc:
            logger.warning("Failed to parse LLM response for relational query: %s", exc)
            return {}

    def generate_comparative_llm(
        self, jsonld_a: str, jsonld_b: str
    ) -> dict:
        """Generate a comparative query using an LLM."""
        prompt = COMPARATIVE_PROMPT.format(jsonld_a=jsonld_a, jsonld_b=jsonld_b)
        response = self.genai_client.models.generate_content(
            model=self.model, contents=prompt
        )
        try:
            result = json.loads(response.text.strip().removeprefix("```json").removesuffix("```"))
            result["type"] = "comparative"
            result["source"] = "llm"
            return result
        except (json.JSONDecodeError, AttributeError) as exc:
            logger.warning("Failed to parse LLM response for comparative query: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_name(entity: dict) -> str | None:
        for key in ("name", "schema:name", "http://schema.org/name"):
            val = entity.get(key)
            if val:
                if isinstance(val, list):
                    val = val[0]
                if isinstance(val, dict):
                    return val.get("@value", "")
                return str(val)
        return None

    @staticmethod
    def _extract_description(entity: dict) -> str | None:
        for key in ("description", "schema:description", "http://schema.org/description"):
            val = entity.get(key)
            if val:
                if isinstance(val, list):
                    val = val[0]
                if isinstance(val, dict):
                    return val.get("@value", "")
                return str(val)
        return None

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def generate_for_domain(self, domain_name: str) -> list[dict]:
        """Generate queries for all entities in a domain."""
        domain_dir = self.raw_dir / domain_name
        if not domain_dir.exists():
            logger.warning("Raw data not found for domain: %s", domain_name)
            return []

        all_queries = []
        entity_data_list = []

        # First pass: load all entities
        for entity_dir in sorted(domain_dir.iterdir()):
            if not entity_dir.is_dir():
                continue
            jsonld_path = entity_dir / "structured_data.json"
            if not jsonld_path.exists():
                jsonld_path = entity_dir / "entity.json"
            if not jsonld_path.exists():
                continue

            data = json.loads(jsonld_path.read_text())
            entities = data if isinstance(data, list) else [data]
            # Filter out non-dict items (some entity files contain nested lists)
            entities = [e for e in entities if isinstance(e, dict)]
            if not entities:
                continue
            entity_data_list.append({"dir": entity_dir, "entities": entities, "raw": data})

        # Second pass: generate queries
        for item in entity_data_list:
            entities = item["entities"]
            main = self._find_main(entities)
            if not main:
                continue

            name = self._extract_name(main) or item["dir"].name
            logger.info("  Generating queries for: %s", name)

            # Links extraction (reused from transformer)
            links = self._extract_links(entities)
            jsonld_str = json.dumps(item["raw"], indent=2, ensure_ascii=False)

            # Factual query
            if self.use_llm:
                fq = self.generate_factual_llm(jsonld_str)
            else:
                fq = self.generate_factual_template(main)
            if fq:
                fq["entity"] = name
                fq["domain"] = domain_name
                all_queries.append(fq)

            # Relational query
            if links:
                if self.use_llm:
                    rq = self.generate_relational_llm(jsonld_str, links)
                else:
                    rq = self.generate_relational_template(main, links)
                if rq:
                    rq["entity"] = name
                    rq["domain"] = domain_name
                    all_queries.append(rq)

        # Comparative queries (pairs within domain)
        if len(entity_data_list) >= 2:
            for i in range(0, len(entity_data_list) - 1, 2):
                a = entity_data_list[i]
                b = entity_data_list[i + 1]
                main_a = self._find_main(a["entities"])
                main_b = self._find_main(b["entities"])
                if not main_a or not main_b:
                    continue
                a_str = json.dumps(a["raw"], indent=2, ensure_ascii=False)
                b_str = json.dumps(b["raw"], indent=2, ensure_ascii=False)
                if self.use_llm:
                    cq = self.generate_comparative_llm(a_str, b_str)
                else:
                    name_a = self._extract_name(main_a) or "A"
                    name_b = self._extract_name(main_b) or "B"
                    cq = {
                        "type": "comparative",
                        "query": f"Compare {name_a} and {name_b}.",
                        "ground_truth": "Requires data from both entities.",
                        "source": "template",
                    }
                if cq:
                    cq["domain"] = domain_name
                    all_queries.append(cq)

        return all_queries

    def _find_main(self, entities: list[dict]) -> dict | None:
        for e in entities:
            if self._extract_name(e):
                types = e.get("@type", [])
                if isinstance(types, str):
                    types = [types]
                type_names = [t.rsplit("/", 1)[-1] for t in types]
                if not any(t in ("WebPage", "Answer", "WebSite", "FAQPage") for t in type_names):
                    return e
        for e in entities:
            if self._extract_name(e):
                return e
        return None

    @staticmethod
    def _extract_links(entities: list[dict]) -> list[dict]:
        links = []
        seen = set()
        props = [
            "provider", "offers", "sameAs", "about", "publisher", "isPartOf",
            "schema:provider", "schema:offers", "schema:sameAs",
            "http://schema.org/provider", "http://schema.org/offers",
            "http://schema.org/sameAs",
        ]
        for e in entities:
            for prop in props:
                val = e.get(prop)
                if not val:
                    continue
                refs = val if isinstance(val, list) else [val]
                for ref in refs:
                    if isinstance(ref, dict) and "@id" in ref:
                        eid = ref["@id"]
                        if eid not in seen:
                            seen.add(eid)
                            label = prop.rsplit("/", 1)[-1]
                            links.append({"url": eid, "relation": label})
        return links


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate test queries from entity data")
    parser.add_argument("--config", type=Path, default=Path("config/experiment_config.yaml"))
    parser.add_argument("--domain", type=str)
    parser.add_argument("--no-llm", action="store_true", help="Use templates only, no LLM")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    generator = QueryGenerator(use_llm=not args.no_llm)

    all_queries = []
    for domain_cfg in config["domains"]:
        name = domain_cfg["name"].lower().replace(" ", "_")
        if args.domain and name != args.domain.lower().replace(" ", "_"):
            continue
        logger.info("=== Generating queries for: %s ===", domain_cfg["name"])
        queries = generator.generate_for_domain(name)
        all_queries.extend(queries)
        logger.info("  Generated %d queries", len(queries))

    # Save all queries
    QUERIES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = QUERIES_DIR / "test_queries.json"
    with open(out_path, "w") as f:
        json.dump(all_queries, f, indent=2, ensure_ascii=False)
    logger.info("Saved %d queries → %s", len(all_queries), out_path)


if __name__ == "__main__":
    main()
