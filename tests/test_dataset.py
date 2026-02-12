"""Tests for dataset collection and transformation modules."""

from __future__ import annotations

import json
import textwrap

import pytest

from src.dataset.transformer import EntityTransformer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_JSONLD = [
    {
        "@id": "https://data.wordlift.io/wl1507823/product/test-product",
        "@type": ["http://schema.org/FinancialProduct", "http://schema.org/Product"],
        "http://schema.org/name": [{"@value": "Test Insurance Product"}],
        "http://schema.org/description": [
            {"@value": "A test insurance product for unit testing."}
        ],
        "http://schema.org/provider": [
            {"@id": "https://data.wordlift.io/wl1507823/organization/test-org"}
        ],
        "http://schema.org/offers": [
            {"@id": "https://data.wordlift.io/wl1507823/product/test-product/offer"}
        ],
        "http://schema.org/category": [{"@value": "insurance"}],
    },
    {
        "@id": "https://data.wordlift.io/wl1507823/organization/test-org",
        "@type": ["http://schema.org/Organization"],
        "http://schema.org/name": [{"@value": "Test Organization"}],
        "http://schema.org/sameAs": [
            {"@id": "https://www.wikidata.org/wiki/Q12345"}
        ],
    },
    {
        "@id": "https://data.wordlift.io/wl1507823/product/test-product/offer",
        "@type": ["http://schema.org/Offer"],
        "http://schema.org/availability": [{"@value": "InStock"}],
    },
    {
        "@type": ["http://schema.org/Question"],
        "@id": "https://data.wordlift.io/test/faq/q/01",
        "http://schema.org/name": [{"@value": "What is this product?"}],
        "http://schema.org/acceptedAnswer": {"@id": "https://data.wordlift.io/test/faq/q/01/a"},
    },
    {
        "@type": ["http://schema.org/Answer"],
        "@id": "https://data.wordlift.io/test/faq/q/01/a",
        "http://schema.org/text": [{"@value": "It is a test insurance product."}],
    },
]


SAMPLE_HTML = textwrap.dedent("""\
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Page</title>
        <script type="application/ld+json">
        {"@type": "Product", "name": "Existing Product"}
        </script>
    </head>
    <body>
        <h1>Test Page</h1>
        <p>Some content here.</p>
    </body>
    </html>
""")


# ---------------------------------------------------------------------------
# Transformer tests
# ---------------------------------------------------------------------------


class TestEntityTransformer:
    """Tests for the EntityTransformer class."""

    def setup_method(self):
        self.transformer = EntityTransformer()

    # -- Variant 1: Plain HTML --

    def test_plain_html_strips_jsonld(self):
        result = self.transformer.create_plain_html(SAMPLE_HTML)
        assert "application/ld+json" not in result
        assert "Existing Product" not in result

    def test_plain_html_preserves_content(self):
        result = self.transformer.create_plain_html(SAMPLE_HTML)
        assert "Test Page" in result
        assert "Some content here" in result

    def test_plain_html_no_jsonld_is_noop(self):
        html = "<html><body><p>No structured data</p></body></html>"
        result = self.transformer.create_plain_html(html)
        assert "No structured data" in result

    # -- Variant 2: HTML + JSON-LD --

    def test_html_with_jsonld_existing_preserved(self):
        result = self.transformer.create_html_with_jsonld(SAMPLE_HTML, SAMPLE_JSONLD)
        assert "Existing Product" in result  # Existing JSON-LD kept

    def test_html_with_jsonld_injected_when_missing(self):
        plain = "<html><head><title>T</title></head><body>Hi</body></html>"
        result = self.transformer.create_html_with_jsonld(plain, SAMPLE_JSONLD)
        assert "application/ld+json" in result
        assert "Test Insurance Product" in result

    # -- Variant 3: Enhanced entity page --

    def test_enhanced_page_contains_entity_name(self):
        result = self.transformer.create_enhanced_entity_page(
            SAMPLE_JSONLD, "https://example.com/entity", "test_domain"
        )
        assert "Test Insurance Product" in result

    def test_enhanced_page_contains_jsonld_block(self):
        result = self.transformer.create_enhanced_entity_page(
            SAMPLE_JSONLD, "https://example.com/entity", "test_domain"
        )
        assert "application/ld+json" in result

    def test_enhanced_page_contains_linked_entities(self):
        result = self.transformer.create_enhanced_entity_page(
            SAMPLE_JSONLD, "https://example.com/entity", "test_domain"
        )
        assert "test-org" in result  # Provider link
        assert "Related Entities" in result

    def test_enhanced_page_contains_agent_instructions(self):
        result = self.transformer.create_enhanced_entity_page(
            SAMPLE_JSONLD, "https://example.com/entity", "test_domain"
        )
        assert "Agent Instructions" in result
        assert "content negotiation" in result.lower()

    def test_enhanced_page_contains_breadcrumb(self):
        result = self.transformer.create_enhanced_entity_page(
            SAMPLE_JSONLD, "https://example.com/entity", "test_domain"
        )
        assert "Thing" in result
        assert "Product" in result

    def test_enhanced_page_contains_faq(self):
        result = self.transformer.create_enhanced_entity_page(
            SAMPLE_JSONLD, "https://example.com/entity", "test_domain"
        )
        assert "What is this product?" in result

    # -- Helper tests --

    def test_get_types_extracts_short_names(self):
        types = EntityTransformer._get_types(SAMPLE_JSONLD[0])
        assert "FinancialProduct" in types
        assert "Product" in types

    def test_extract_links_finds_provider(self):
        links = EntityTransformer._extract_links(SAMPLE_JSONLD)
        urls = [l["url"] for l in links]
        assert any("test-org" in u for u in urls)

    def test_extract_links_finds_sameas(self):
        links = EntityTransformer._extract_links(SAMPLE_JSONLD)
        urls = [l["url"] for l in links]
        assert any("wikidata" in u for u in urls)

    def test_find_main_entity_skips_webpage(self):
        entities = [
            {"@type": ["http://schema.org/WebPage"], "http://schema.org/name": [{"@value": "Page"}]},
            {"@type": ["http://schema.org/Product"], "http://schema.org/name": [{"@value": "Prod"}]},
        ]
        main = self.transformer._find_main_entity(entities)
        types = EntityTransformer._get_types(main)
        assert "Product" in types
