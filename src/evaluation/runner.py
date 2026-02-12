"""Experiment runner — orchestrates all conditions and queries.

Runs the full experiment matrix (6 conditions × N queries),
saves raw results, and handles rate limiting / checkpointing.

Supports incremental per-query checkpointing and resume:
  - Each query result is appended to a JSONL checkpoint file immediately.
  - On restart, completed queries are detected and skipped.
  - Safe to stop at any time — progress is never lost.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import yaml

from config.settings import CONDITIONS, DATA_DIR, QUERIES_DIR, RESULTS_DIR
from src.evaluation.metrics import MetricsCalculator
from src.indexing.vectorsearch import VectorSearchManager
from src.retrieval.agentic_rag import AgenticRAGPipeline
from src.retrieval.standard_rag import StandardRAGPipeline

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Orchestrates the full experiment across all conditions."""

    def __init__(
        self,
        config_path: Path = Path("config/experiment_config.yaml"),
        results_dir: Path = RESULTS_DIR,
    ) -> None:
        self.results_dir = results_dir / "raw"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.vector_search = VectorSearchManager()
        self.standard_rag = StandardRAGPipeline(vector_search=self.vector_search)
        self.agentic_rag = AgenticRAGPipeline(vector_search=self.vector_search)
        self.metrics = MetricsCalculator()

        # Build domain name → account_id mapping from domains.json
        self._domain_account_map = {}
        domains_path = DATA_DIR / "domains.json"
        if domains_path.exists():
            with open(domains_path) as f:
                domains_data = json.load(f)
                for d in domains_data.get("domains", domains_data):
                    if isinstance(d, dict):
                        # Convert name to snake_case key (e.g. "WordLift Blog" → "wordlift_blog")
                        name_key = d["name"].lower().replace(" ", "_")
                        self._domain_account_map[name_key] = d.get("account_id", "")
        logger.info("Domain-account map: %s", self._domain_account_map)

    def load_queries(self, queries_path: Path | None = None) -> list[dict]:
        """Load test queries from the generated query file."""
        path = queries_path or (QUERIES_DIR / "test_queries.json")
        if not path.exists():
            logger.error("Queries file not found: %s", path)
            return []
        with open(path) as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _checkpoint_path(self, condition: str) -> Path:
        """Path for the JSONL checkpoint file for a condition."""
        return self.results_dir / f"{condition}_checkpoint.jsonl"

    def _load_checkpoint(self, condition: str) -> list[dict]:
        """Load existing checkpoint results for a condition."""
        path = self._checkpoint_path(condition)
        if not path.exists():
            return []
        results = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return results

    def _completed_queries(self, condition: str) -> set[str]:
        """Return set of query strings already completed for this condition."""
        checkpoint = self._load_checkpoint(condition)
        return {r["query"] for r in checkpoint if "query" in r}

    def _append_checkpoint(self, condition: str, result: dict) -> None:
        """Append a single result to the checkpoint file."""
        path = self._checkpoint_path(condition)
        with open(path, "a") as f:
            f.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")

    # ------------------------------------------------------------------
    # Run a single condition
    # ------------------------------------------------------------------

    def run_condition(
        self,
        condition: str,
        queries: list[dict],
        evaluate: bool = True,
    ) -> list[dict]:
        """Run all queries for a single experimental condition.

        Supports resume: skips queries that exist in the checkpoint file.

        Args:
            condition: Condition ID (C1-C6).
            queries: List of query dicts.
            evaluate: Whether to compute metrics (slower).

        Returns:
            List of result dicts with answers and metrics.
        """
        cond_cfg = self.config["conditions"].get(condition, {})
        retrieval_mode = cond_cfg.get("retrieval_mode", "standard")
        doc_format = cond_cfg.get("document_format", "")

        # Load checkpoint for resume
        completed = self._completed_queries(condition)
        existing_results = self._load_checkpoint(condition)

        logger.info(
            "=== Running condition %s: %s ===",
            condition,
            cond_cfg.get("description", ""),
        )
        if completed:
            logger.info("  Resuming: %d/%d queries already completed", len(completed), len(queries))

        results = list(existing_results)  # Start from checkpoint

        for i, query_data in enumerate(queries):
            # Skip already completed queries
            if query_data["query"] in completed:
                continue

            logger.info(
                "  Query %d/%d [%s]: %s",
                i + 1,
                len(queries),
                query_data.get("type", "?"),
                query_data["query"][:80],
            )

            try:
                if retrieval_mode == "agentic":
                    # Set the domain account_id for this query's KG
                    query_domain = query_data.get("domain", "")
                    self.agentic_rag.domain_account_id = self._domain_account_map.get(
                        query_domain, ""
                    )
                    rag_result = self.agentic_rag.query(query_data["query"], condition)
                    result = {
                        "query": rag_result.query,
                        "answer": rag_result.answer,
                        "ground_truth": query_data.get("ground_truth", ""),
                        "retrieved_documents": rag_result.retrieved_documents,
                        "condition": condition,
                        "retrieval_mode": "agentic",
                        "document_format": doc_format,
                        "query_type": query_data.get("type", "unknown"),
                        "domain": query_data.get("domain", ""),
                        "entity": query_data.get("entity", ""),
                        # Agentic-specific
                        "links_followed": rag_result.links_followed,
                        "links_available": rag_result.links_available,
                        "max_hop_depth": rag_result.max_hop_depth,
                        "agent_steps": [
                            {"tool": s.tool, "hop_depth": s.hop_depth}
                            for s in rag_result.agent_steps
                        ],
                    }
                else:
                    rag_result = self.standard_rag.query(query_data["query"], condition)
                    result = {
                        "query": rag_result.query,
                        "answer": rag_result.answer,
                        "ground_truth": query_data.get("ground_truth", ""),
                        "retrieved_documents": rag_result.retrieved_documents,
                        "condition": condition,
                        "retrieval_mode": "standard",
                        "document_format": doc_format,
                        "query_type": query_data.get("type", "unknown"),
                        "domain": query_data.get("domain", ""),
                        "entity": query_data.get("entity", ""),
                    }

                # Evaluate
                if evaluate:
                    metrics = self.metrics.evaluate_result(result)
                    result["metrics"] = metrics

                results.append(result)

                # Checkpoint immediately
                self._append_checkpoint(condition, result)

            except Exception as exc:
                logger.error("  Error on query %d: %s", i + 1, exc)
                error_result = {
                    "query": query_data["query"],
                    "error": str(exc),
                    "condition": condition,
                    "query_type": query_data.get("type", "unknown"),
                }
                results.append(error_result)
                self._append_checkpoint(condition, error_result)

            # Rate limiting
            time.sleep(0.5)

        return results

    def run_all(
        self,
        queries: list[dict] | None = None,
        conditions: list[str] | None = None,
        evaluate: bool = True,
    ) -> dict[str, list[dict]]:
        """Run all experimental conditions.

        Args:
            queries: Optional list of queries. Loads from file if None.
            conditions: Optional list of condition IDs. Runs all if None.
            evaluate: Whether to compute metrics.

        Returns:
            Dict mapping condition → list of results.
        """
        if queries is None:
            queries = self.load_queries()
            if not queries:
                logger.error("No queries found. Run query_generator first.")
                return {}

        target_conditions = conditions or list(CONDITIONS.keys())
        all_results = {}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for condition in target_conditions:
            results = self.run_condition(condition, queries, evaluate=evaluate)
            all_results[condition] = results

            # Save final condition results (clean JSON)
            out_path = self.results_dir / f"{condition}_{timestamp}.json"
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            logger.info("  Results saved → %s", out_path)

        # Save combined results
        combined_path = self.results_dir / f"all_results_{timestamp}.json"
        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        logger.info("Combined results → %s", combined_path)

        return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiment conditions")
    parser.add_argument("--config", type=Path, default=Path("config/experiment_config.yaml"))
    parser.add_argument("--conditions", nargs="+", help="Conditions to run (e.g. C1 C2)")
    parser.add_argument("--queries", type=Path, help="Path to queries JSON")
    parser.add_argument("--no-eval", action="store_true", help="Skip metric evaluation")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without executing")
    parser.add_argument(
        "--reset", nargs="*", metavar="COND",
        help="Clear checkpoint for specified conditions (or all if none given) and re-run from scratch",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    runner = ExperimentRunner(config_path=args.config)

    # Handle --reset
    if args.reset is not None:
        to_reset = args.reset if args.reset else list(CONDITIONS.keys())
        for c in to_reset:
            cp = runner._checkpoint_path(c)
            if cp.exists():
                cp.unlink()
                logger.info("Cleared checkpoint for %s", c)

    queries = runner.load_queries(args.queries)
    logger.info("Loaded %d queries", len(queries))

    if args.dry_run:
        conditions = args.conditions or list(CONDITIONS.keys())
        print(f"\nDry run — would execute:")
        print(f"  Conditions: {conditions}")
        print(f"  Queries: {len(queries)}")
        print(f"  Total runs: {len(conditions) * len(queries)}")
        for c in conditions:
            cfg = runner.config["conditions"].get(c, {})
            completed = len(runner._completed_queries(c))
            remaining = len(queries) - completed
            print(f"  {c}: {cfg.get('description', '?')} — {completed} done, {remaining} remaining")
        return

    results = runner.run_all(
        queries=queries,
        conditions=args.conditions,
        evaluate=not args.no_eval,
    )

    # Summary
    print("\n=== Experiment Summary ===")
    for cond, res in results.items():
        errors = sum(1 for r in res if "error" in r)
        print(f"  {cond}: {len(res)} queries, {errors} errors")


if __name__ == "__main__":
    main()
