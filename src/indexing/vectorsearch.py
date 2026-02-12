"""Vertex AI Vector Search 2.0 integration.

Uses the real vectorsearch_v1beta SDK following the patterns from:
  https://medium.com/google-cloud/10-minute-agentic-rag-with-the-new-vector-search-2-0-and-adk

Handles:
  - Collection creation with auto-embeddings (gemini-embedding-001)
  - Document ingestion as Data Objects
  - Hybrid search (semantic + keyword) with Reciprocal Rank Fusion
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from config.settings import (
    COLLECTION_PREFIX,
    GCP_PROJECT_ID,
    GCP_REGION,
    PROCESSED_DIR,
    TOP_K,
)

logger = logging.getLogger(__name__)

# Map conditions → document variant filename
FORMAT_MAP = {
    "C1": "plain.html",
    "C2": "with_jsonld.html",
    "C3": "enhanced.html",
    "C4": "plain.html",
    "C5": "with_jsonld.html",
    "C6": "enhanced.html",
    "C6_PLUS": "enhanced_plus.html",
}

# Map document format → collection suffix (3 collections, not 6)
# NOTE: VS2.0 requires collection IDs: lowercase letters, digits, hyphens only
FORMAT_TO_COLLECTION = {
    "plain.html": "plain-html",
    "with_jsonld.html": "html-jsonld",
    "enhanced.html": "enhanced",
    "enhanced_plus.html": "enhanced-plus",
}

# Embedding config
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768


class VectorSearchManager:
    """Manages Vertex AI Vector Search 2.0 collections for the experiment.

    Creates 3 collections (one per document format):
      - {prefix}_plain_html     → used by C1, C4
      - {prefix}_html_jsonld    → used by C2, C5
      - {prefix}_enhanced       → used by C3, C6

    Retrieval mode (standard vs agentic) is handled by the RAG pipelines,
    not the collection structure.
    """

    def __init__(
        self,
        project_id: str = GCP_PROJECT_ID,
        region: str = GCP_REGION,
    ) -> None:
        self.project_id = project_id
        self.region = region
        self.parent = f"projects/{project_id}/locations/{region}"
        self._admin_client = None
        self._data_client = None
        self._search_client = None

    # ------------------------------------------------------------------
    # Lazy-init clients
    # ------------------------------------------------------------------

    @property
    def admin_client(self):
        """VectorSearchServiceClient for collection management."""
        if self._admin_client is None:
            from google.cloud import vectorsearch_v1beta
            self._admin_client = vectorsearch_v1beta.VectorSearchServiceClient()
        return self._admin_client

    @property
    def data_client(self):
        """DataObjectServiceClient for data ingestion."""
        if self._data_client is None:
            from google.cloud import vectorsearch_v1beta
            self._data_client = vectorsearch_v1beta.DataObjectServiceClient()
        return self._data_client

    @property
    def search_client(self):
        """DataObjectSearchServiceClient for search queries."""
        if self._search_client is None:
            from google.cloud import vectorsearch_v1beta
            self._search_client = vectorsearch_v1beta.DataObjectSearchServiceClient()
        return self._search_client

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def get_collection_name(self, condition: str) -> str:
        """Get the collection name for an experimental condition."""
        filename = FORMAT_MAP.get(condition, "enhanced.html")
        suffix = FORMAT_TO_COLLECTION[filename]
        return f"{COLLECTION_PREFIX}-{suffix}"

    def get_collection_path(self, condition: str) -> str:
        """Get the full resource path for a collection."""
        name = self.get_collection_name(condition)
        return f"{self.parent}/collections/{name}"

    def create_collection(self, doc_format: str) -> str:
        """Create a Vector Search 2.0 collection for a document format.

        Uses auto-embeddings with gemini-embedding-001 so we can ingest
        raw HTML text without a separate embedding pipeline.

        Args:
            doc_format: One of 'plain_html', 'html_jsonld', 'enhanced'.

        Returns:
            The collection resource name.
        """
        from google.cloud import vectorsearch_v1beta

        collection_id = f"{COLLECTION_PREFIX}-{doc_format}"
        logger.info("Creating collection: %s", collection_id)

        # Data schema: each document has content + metadata fields
        data_schema = {
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "domain": {"type": "string"},
                "entity": {"type": "string"},
                "doc_format": {"type": "string"},
            },
        }

        # Vector schema: auto-embed the content field with gemini-embedding-001
        vector_schema = {
            "content_embedding": {
                "dense_vector": {
                    "dimensions": EMBEDDING_DIMENSIONS,
                    "vertex_embedding_config": {
                        "model_id": EMBEDDING_MODEL,
                        "text_template": "{content}",
                        "task_type": "RETRIEVAL_DOCUMENT",
                    },
                },
            },
        }

        collection = vectorsearch_v1beta.Collection(
            data_schema=data_schema,
            vector_schema=vector_schema,
        )

        request = vectorsearch_v1beta.CreateCollectionRequest(
            parent=self.parent,
            collection_id=collection_id,
            collection=collection,
        )

        try:
            operation = self.admin_client.create_collection(request=request)
            result = operation.result()
            logger.info("  ✓ Collection created: %s", collection_id)
            return f"{self.parent}/collections/{collection_id}"
        except Exception as exc:
            # Collection might already exist
            if "already exists" in str(exc).lower() or "ALREADY_EXISTS" in str(exc):
                logger.info("  Collection already exists: %s", collection_id)
                return f"{self.parent}/collections/{collection_id}"
            raise

    def setup_all_collections(self) -> dict[str, str]:
        """Create all 3 collections (one per document format)."""
        results = {}
        for doc_format in FORMAT_TO_COLLECTION.values():
            name = self.create_collection(doc_format)
            results[doc_format] = name
        return results

    def delete_collection(self, doc_format: str) -> None:
        """Delete a collection by format name."""
        from google.cloud import vectorsearch_v1beta

        collection_id = f"{COLLECTION_PREFIX}-{doc_format}"
        name = f"{self.parent}/collections/{collection_id}"
        logger.info("Deleting collection: %s", collection_id)
        try:
            request = vectorsearch_v1beta.DeleteCollectionRequest(name=name)
            operation = self.admin_client.delete_collection(request=request)
            operation.result()
            logger.info("  ✓ Deleted")
        except Exception as exc:
            logger.warning("  Could not delete: %s", exc)

    # ------------------------------------------------------------------
    # Document ingestion
    # ------------------------------------------------------------------

    def ingest_documents(
        self,
        condition: str,
        documents: list[dict],
        batch_size: int = 50,
    ) -> int:
        """Ingest documents into a Vector Search 2.0 collection.

        Each document dict should have:
          - id: unique document identifier
          - content: text content (auto-embedded by VS2.0)
          - metadata: dict with domain, entity, format

        Args:
            condition: Experimental condition (C1-C6).
            documents: List of document dicts.
            batch_size: Docs per progress update.

        Returns:
            Number of documents ingested.
        """
        from google.cloud import vectorsearch_v1beta

        collection_path = self.get_collection_path(condition)
        logger.info(
            "Ingesting %d documents into %s", len(documents), collection_path
        )

        ingested = 0
        errors = 0

        for i, doc in enumerate(documents):
            # Sanitize data_object_id: lowercase letters, digits, hyphens only
            doc_id = self._sanitize_id(doc["id"])

            # Truncate content to stay within embedding token limits
            # gemini-embedding-001 has ~8k token limit; be conservative
            content = doc["content"]
            if len(content) > 20000:
                content = content[:20000]

            data_object = vectorsearch_v1beta.DataObject(
                data={
                    "content": content,
                    "domain": doc["metadata"]["domain"],
                    "entity": doc["metadata"]["entity"],
                    "doc_format": doc["metadata"]["format"],
                },
                # Empty vectors → auto-embedding generation
                vectors={},
            )

            request = vectorsearch_v1beta.CreateDataObjectRequest(
                parent=collection_path,
                data_object_id=doc_id,
                data_object=data_object,
            )

            try:
                self.data_client.create_data_object(request=request)
                ingested += 1
            except Exception as exc:
                err_str = str(exc)
                if "already exists" in err_str.lower():
                    ingested += 1  # Count as success
                    logger.debug("  Skipping existing: %s", doc_id)
                else:
                    errors += 1
                    logger.error("  Failed [%s]: %s", doc_id, err_str[:200])

            # Progress logging
            if (i + 1) % batch_size == 0 or i == len(documents) - 1:
                logger.info(
                    "  Progress: %d/%d ingested, %d errors",
                    ingested, len(documents), errors,
                )

            # Light rate limiting every 10 docs
            if (i + 1) % 10 == 0:
                time.sleep(0.5)

        logger.info(
            "  ✓ Ingested %d / %d documents (%d errors)",
            ingested, len(documents), errors,
        )
        return ingested

    @staticmethod
    def _sanitize_id(raw_id: str) -> str:
        """Sanitize an ID to meet VS2.0 data_object_id constraints.

        Rules: lowercase letters, digits, hyphens only; start with letter;
               end with letter or digit; max 64 chars.
        """
        import hashlib
        import re
        # Replace slashes, underscores, dots with hyphens
        sanitized = raw_id.lower().replace("/", "-").replace("_", "-").replace(".", "-")
        # Remove any remaining invalid characters
        sanitized = re.sub(r"[^a-z0-9-]", "", sanitized)
        # Collapse multiple hyphens
        sanitized = re.sub(r"-+", "-", sanitized)
        # Ensure starts with letter
        sanitized = sanitized.lstrip("-0123456789")
        # Ensure ends with letter/digit
        sanitized = sanitized.rstrip("-")
        # Truncate to 64 chars; use a hash suffix for uniqueness if too long
        if len(sanitized) > 64:
            hash_suffix = hashlib.md5(raw_id.encode()).hexdigest()[:8]
            # 64 - 1 (hyphen) - 8 (hash) = 55 chars for prefix
            sanitized = sanitized[:55].rstrip("-") + "-" + hash_suffix
        return sanitized or "doc-unknown"

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        condition: str,
        query: str,
        top_k: int = TOP_K,
        hybrid: bool = True,
        filter_dict: dict | None = None,
    ) -> list[dict]:
        """Search a collection using hybrid search (semantic + keyword).

        Follows the article pattern: BatchSearchDataObjectsRequest with
        SemanticSearch + TextSearch combined via Reciprocal Rank Fusion.

        Args:
            condition: Experimental condition (C1-C6).
            query: Natural language query.
            top_k: Number of results to return.
            hybrid: Whether to use hybrid search (semantic + keyword).
            filter_dict: Optional metadata filter dict.

        Returns:
            List of result dicts with id, content, score, metadata.
        """
        from google.cloud import vectorsearch_v1beta

        collection_path = self.get_collection_path(condition)
        logger.debug(
            "Searching %s: '%s' (top_k=%d, hybrid=%s)",
            collection_path, query[:80], top_k, hybrid,
        )

        output_fields = vectorsearch_v1beta.OutputFields(
            data_fields=["content", "domain", "entity", "doc_format"]
        )

        # Build semantic search kwargs (only include filter if provided)
        semantic_kwargs = {
            "search_text": query,
            "search_field": "content_embedding",
            "task_type": "QUESTION_ANSWERING",  # Pairs with RETRIEVAL_DOCUMENT
            "top_k": top_k,
            "output_fields": output_fields,
        }
        if filter_dict:
            semantic_kwargs["filter"] = filter_dict

        semantic_search = vectorsearch_v1beta.SemanticSearch(**semantic_kwargs)
        searches = [vectorsearch_v1beta.Search(semantic_search=semantic_search)]

        if hybrid:
            # Text search — keyword matching on content field
            text_kwargs = {
                "search_text": query,
                "data_field_names": ["content"],
                "top_k": top_k,
                "output_fields": output_fields,
            }
            if filter_dict:
                text_kwargs["filter"] = filter_dict

            text_search = vectorsearch_v1beta.TextSearch(**text_kwargs)
            searches.append(vectorsearch_v1beta.Search(text_search=text_search))

        # Build the search request
        request_kwargs = {
            "parent": collection_path,
            "searches": searches,
        }

        # Add RRF ranking if hybrid
        if hybrid and len(searches) > 1:
            request_kwargs["combine"] = (
                vectorsearch_v1beta.BatchSearchDataObjectsRequest.CombineResultsOptions(
                    ranker=vectorsearch_v1beta.Ranker(
                        rrf=vectorsearch_v1beta.ReciprocalRankFusion(
                            weights=[0.7, 0.3]  # semantic-heavy for our use case
                        )
                    )
                )
            )

        request = vectorsearch_v1beta.BatchSearchDataObjectsRequest(**request_kwargs)
        response = self.search_client.batch_search_data_objects(request=request)

        # Parse results
        results = []
        if response.results:
            for res in response.results[0].results:
                data = res.data_object.data if res.data_object else {}
                results.append({
                    "id": res.data_object.name if res.data_object else "",
                    "content": data.get("content", ""),
                    "score": getattr(res, "score", 0.0),
                    "metadata": {
                        "domain": data.get("domain", ""),
                        "entity": data.get("entity", ""),
                        "format": data.get("doc_format", ""),
                    },
                })

        return results

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def load_documents_for_condition(
        self,
        condition: str,
        processed_dir: Path = PROCESSED_DIR,
    ) -> list[dict]:
        """Load processed documents appropriate for a condition.

        Maps conditions to document variants:
          C1, C4 → plain.html
          C2, C5 → with_jsonld.html
          C3, C6 → enhanced.html
        """
        filename = FORMAT_MAP.get(condition, "enhanced.html")
        documents = []

        for domain_dir in sorted(processed_dir.iterdir()):
            if not domain_dir.is_dir():
                continue
            for entity_dir in sorted(domain_dir.iterdir()):
                if not entity_dir.is_dir():
                    continue
                doc_path = entity_dir / filename
                if doc_path.exists():
                    documents.append({
                        "id": f"{domain_dir.name}/{entity_dir.name}",
                        "content": doc_path.read_text(errors="replace"),
                        "metadata": {
                            "domain": domain_dir.name,
                            "entity": entity_dir.name,
                            "format": filename,
                            "condition": condition,
                        },
                    })

        return documents

    def ingest_for_condition(
        self, condition: str, processed_dir: Path = PROCESSED_DIR
    ) -> int:
        """Load and ingest documents for a given condition."""
        docs = self.load_documents_for_condition(condition, processed_dir)
        logger.info("Loaded %d documents for %s", len(docs), condition)
        if docs:
            return self.ingest_documents(condition, docs)
        return 0

    def ingest_all(self, processed_dir: Path = PROCESSED_DIR) -> dict[str, int]:
        """Ingest documents for all 3 unique document formats.

        Since C1/C4, C2/C5, C3/C6 share collections, we only ingest 3 times.
        """
        results = {}
        for condition in ("C1", "C2", "C3"):  # Only the unique formats
            count = self.ingest_for_condition(condition, processed_dir)
            results[condition] = count
        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vertex AI Vector Search 2.0 management"
    )
    parser.add_argument(
        "--setup", action="store_true", help="Create all 3 collections"
    )
    parser.add_argument(
        "--ingest", type=str, help="Ingest documents for condition (e.g. C1) or 'all'"
    )
    parser.add_argument(
        "--search", type=str, help="Test search query"
    )
    parser.add_argument(
        "--condition", type=str, default="C2",
        help="Condition for search/ingest (default: C2)"
    )
    parser.add_argument(
        "--delete", type=str,
        help="Delete a collection by format (plain_html, html_jsonld, enhanced)"
    )
    parser.add_argument(
        "--delete-all", action="store_true", help="Delete all collections"
    )
    parser.add_argument(
        "--test", action="store_true", help="Test connection"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    manager = VectorSearchManager()

    if args.setup:
        results = manager.setup_all_collections()
        for fmt, name in results.items():
            logger.info("%s → %s", fmt, name)

    if args.ingest:
        if args.ingest.lower() == "all":
            results = manager.ingest_all()
            for cond, count in results.items():
                logger.info("%s: ingested %d documents", cond, count)
        else:
            count = manager.ingest_for_condition(args.ingest)
            logger.info("Ingested %d documents for %s", count, args.ingest)

    if args.search:
        results = manager.search(args.condition, args.search)
        logger.info("Found %d results", len(results))
        for r in results[:3]:
            logger.info(
                "  [%s] %s / %s (score=%.3f)",
                r["metadata"]["format"],
                r["metadata"]["domain"],
                r["metadata"]["entity"],
                r.get("score", 0),
            )

    if args.delete:
        manager.delete_collection(args.delete)

    if args.delete_all:
        for fmt in FORMAT_TO_COLLECTION.values():
            manager.delete_collection(fmt)

    if args.test:
        logger.info("Testing Vector Search 2.0 connection...")
        logger.info("  Project: %s", manager.project_id)
        logger.info("  Region: %s", manager.region)
        logger.info("  Parent: %s", manager.parent)
        # Try listing collections
        try:
            from google.cloud import vectorsearch_v1beta
            request = vectorsearch_v1beta.ListCollectionsRequest(
                parent=manager.parent
            )
            response = manager.admin_client.list_collections(request=request)
            collections = list(response)
            logger.info("  Found %d existing collections", len(collections))
            for c in collections:
                logger.info("    - %s", c.name)
            logger.info("✓ Connection test passed")
        except Exception as exc:
            logger.error("  Connection test failed: %s", exc)


if __name__ == "__main__":
    main()
