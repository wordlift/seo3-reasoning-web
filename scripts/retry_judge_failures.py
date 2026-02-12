"""Retry LLM judge calls for results where the judge failed.

Loads the deduplicated combined results, identifies entries where
accuracy_reasoning == "Judge failed", re-runs the accuracy and
completeness judge calls, and saves an updated results file.
"""

import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.metrics import MetricsCalculator

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    results_path = Path("results/raw/all_results_combined.json")
    output_path = Path("results/raw/all_results_repaired.json")

    with open(results_path) as f:
        data = json.load(f)

    calculator = MetricsCalculator()

    total_retried = 0
    total_fixed = 0
    total_still_failed = 0

    for condition in sorted(data.keys()):
        results = data[condition]
        failures = []
        for i, r in enumerate(results):
            if "error" in r:
                continue
            m = r.get("metrics", {})
            if m.get("accuracy_reasoning") == "Judge failed" or (
                m.get("accuracy_score", None) == 0
                and m.get("completeness_score", None) == 0
            ):
                failures.append(i)

        if not failures:
            logger.info(f"{condition}: no judge failures, skipping")
            continue

        logger.info(f"{condition}: retrying {len(failures)} judge failures...")

        for idx in failures:
            r = results[idx]
            query = r.get("query", "")
            answer = r.get("answer", "")
            ground_truth = r.get("ground_truth", "")

            if not ground_truth or not answer:
                logger.warning(f"  Skipping query (no GT/answer): {query[:60]}")
                continue

            total_retried += 1

            # Retry accuracy
            acc = calculator.score_accuracy(query, ground_truth, answer)
            new_acc_score = acc.get("score", 0)
            new_acc_reasoning = acc.get("reasoning", "Judge failed")

            # Retry completeness
            comp = calculator.score_completeness(query, ground_truth, answer)
            new_comp_score = comp.get("score", 0)

            if new_acc_reasoning == "Judge failed" or new_acc_score == 0:
                # Still failed — try once more after a brief pause
                time.sleep(1)
                acc = calculator.score_accuracy(query, ground_truth, answer)
                new_acc_score = acc.get("score", 0)
                new_acc_reasoning = acc.get("reasoning", "Judge failed")
                comp = calculator.score_completeness(query, ground_truth, answer)
                new_comp_score = comp.get("score", 0)

            old_acc = r["metrics"].get("accuracy_score", 0)
            old_comp = r["metrics"].get("completeness_score", 0)

            if new_acc_reasoning != "Judge failed" and new_acc_score > 0:
                r["metrics"]["accuracy_score"] = new_acc_score
                r["metrics"]["accuracy_reasoning"] = new_acc_reasoning
                r["metrics"]["completeness_score"] = new_comp_score
                r["metrics"]["facts_covered"] = comp.get("facts_covered", [])
                r["metrics"]["facts_missing"] = comp.get("facts_missing", [])
                total_fixed += 1
                logger.info(
                    f"  ✓ Fixed [{condition}] {query[:50]}... "
                    f"acc: {old_acc}→{new_acc_score}, comp: {old_comp}→{new_comp_score}"
                )
            else:
                total_still_failed += 1
                logger.warning(
                    f"  ✗ Still failed [{condition}] {query[:50]}..."
                )

            # Throttle to avoid rate limits
            time.sleep(0.3)

    # Save repaired results
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"\n=== Repair Summary ===")
    logger.info(f"  Total retried: {total_retried}")
    logger.info(f"  Fixed: {total_fixed}")
    logger.info(f"  Still failed: {total_still_failed}")
    logger.info(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()
