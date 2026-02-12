"""Statistical analysis and visualization of experiment results.

Performs:
  - Paired t-tests with Bonferroni correction
  - Cohen's d effect sizes
  - Confidence intervals
  - Comparison tables (LaTeX-ready)
  - Visualization (bar charts, heatmaps)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from config.settings import RESULTS_DIR

logger = logging.getLogger(__name__)

# Plotting style
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["figure.dpi"] = 150


class ResultsAnalyzer:
    """Analyzes experiment results and generates figures/tables."""

    def __init__(
        self,
        results_dir: Path = RESULTS_DIR / "raw",
        output_dir: Path = RESULTS_DIR,
    ) -> None:
        self.results_dir = results_dir
        self.figures_dir = output_dir / "figures"
        self.tables_dir = output_dir / "tables"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)

    def load_results(self, pattern: str = "all_results_*.json") -> dict:
        """Load the most recent combined results file."""
        files = sorted(self.results_dir.glob(pattern))
        if not files:
            logger.error("No results found matching %s", pattern)
            return {}
        latest = files[-1]
        logger.info("Loading results from %s", latest)
        with open(latest) as f:
            return json.load(f)

    def results_to_dataframe(self, results: dict) -> pd.DataFrame:
        """Convert results dict to a flat DataFrame for analysis."""
        rows = []
        for condition, condition_results in results.items():
            for r in condition_results:
                if "error" in r:
                    continue
                metrics = r.get("metrics", {})
                # Agentic metrics may be at top level or inside metrics
                links_followed = r.get("links_followed", metrics.get("links_followed", []))
                links_available = r.get("links_available", metrics.get("links_available", []))
                agent_steps = r.get("agent_steps", [])
                row = {
                    "condition": condition,
                    "query": r.get("query", ""),
                    "query_type": r.get("query_type", ""),
                    "domain": r.get("domain", ""),
                    "retrieval_mode": r.get("retrieval_mode", ""),
                    "document_format": r.get("document_format", ""),
                    "accuracy": metrics.get("accuracy_score", np.nan),
                    "completeness": metrics.get("completeness_score", np.nan),
                    "grounding": metrics.get("grounding_score", np.nan),
                    "citation_accuracy": metrics.get("citation_accuracy", np.nan),
                    "link_utilization": (
                        len(links_followed) / max(len(links_available), 1)
                        if isinstance(links_followed, list)
                        else metrics.get("link_utilization", np.nan)
                    ),
                    "max_hop_depth": r.get("max_hop_depth", metrics.get("max_hop_depth", np.nan)),
                    "num_tool_calls": (
                        len(agent_steps) if agent_steps
                        else metrics.get("num_tool_calls", np.nan)
                    ),
                }
                rows.append(row)

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Statistical tests
    # ------------------------------------------------------------------

    @staticmethod
    def paired_t_test(
        group_a: np.ndarray, group_b: np.ndarray
    ) -> dict:
        """Paired t-test between two groups with effect size."""
        # Drop NaN pairs
        mask = ~(np.isnan(group_a) | np.isnan(group_b))
        a, b = group_a[mask], group_b[mask]

        if len(a) < 3:
            return {"t_stat": np.nan, "p_value": np.nan, "cohens_d": np.nan, "n": len(a)}

        t_stat, p_value = stats.ttest_rel(a, b)

        # Cohen's d for paired samples
        diff = a - b
        cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0.0

        return {
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "cohens_d": float(cohens_d),
            "mean_a": float(np.mean(a)),
            "mean_b": float(np.mean(b)),
            "std_a": float(np.std(a, ddof=1)),
            "std_b": float(np.std(b, ddof=1)),
            "n": int(len(a)),
            "ci_95_diff": (
                float(np.mean(diff) - 1.96 * np.std(diff, ddof=1) / np.sqrt(len(diff))),
                float(np.mean(diff) + 1.96 * np.std(diff, ddof=1) / np.sqrt(len(diff))),
            ),
        }

    def hypothesis_tests(self, df: pd.DataFrame) -> dict:
        """Run the 3 hypothesis tests with Bonferroni correction.

        H1: C2 vs C1 (structured data helps standard RAG)
        H2: C5 vs C2 (agentic RAG helps beyond standard)
        H3: C6 vs C5 (enhanced pages help agentic RAG)
        """
        comparisons = {
            "H1_structured_data": ("C1", "C2"),
            "H2_agentic_rag": ("C2", "C5"),
            "H3_enhanced_pages": ("C5", "C6"),
        }

        metrics_to_test = ["accuracy", "completeness", "grounding"]
        results = {}
        n_tests = len(comparisons) * len(metrics_to_test)

        for hyp_name, (cond_a, cond_b) in comparisons.items():
            results[hyp_name] = {}
            for metric in metrics_to_test:
                # Merge on query to ensure paired alignment
                df_a = df[df["condition"] == cond_a][["query", metric]].rename(columns={metric: "a"})
                df_b = df[df["condition"] == cond_b][["query", metric]].rename(columns={metric: "b"})
                merged = df_a.merge(df_b, on="query", how="inner")
                a = merged["a"].values
                b = merged["b"].values

                test = self.paired_t_test(a, b)
                # Bonferroni correction
                test["p_value_corrected"] = min(test["p_value"] * n_tests, 1.0)
                test["significant"] = test["p_value_corrected"] < 0.05

                results[hyp_name][metric] = test

        return results

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot_condition_comparison(self, df: pd.DataFrame) -> Path:
        """Bar chart comparing all conditions on key metrics."""
        metrics = ["accuracy", "completeness", "grounding"]
        fig, axes = plt.subplots(1, len(metrics), figsize=(18, 6))

        condition_order = ["C1", "C2", "C3", "C4", "C5", "C6"]
        colors = {
            "C1": "#95a5a6", "C2": "#3498db", "C3": "#2ecc71",
            "C4": "#e67e22", "C5": "#e74c3c", "C6": "#9b59b6",
        }

        for i, metric in enumerate(metrics):
            ax = axes[i]
            means = df.groupby("condition")[metric].mean().reindex(condition_order)
            stds = df.groupby("condition")[metric].std().reindex(condition_order)

            bars = ax.bar(
                means.index,
                means.values,
                yerr=stds.values,
                color=[colors.get(c, "#gray") for c in means.index],
                capsize=4,
                alpha=0.85,
            )
            ax.set_title(metric.replace("_", " ").title(), fontweight="bold")
            ax.set_ylabel("Score")
            ax.set_ylim(0, 5.5 if metric in ("accuracy", "completeness") else 1.1)

        plt.tight_layout()
        path = self.figures_dir / "condition_comparison.png"
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        logger.info("Saved → %s", path)
        return path

    def plot_query_type_breakdown(self, df: pd.DataFrame) -> Path:
        """Heatmap of accuracy by condition × query type."""
        pivot = df.pivot_table(
            values="accuracy",
            index="query_type",
            columns="condition",
            aggfunc="mean",
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            vmin=1,
            vmax=5,
            ax=ax,
            linewidths=0.5,
        )
        ax.set_title("Answer Accuracy by Query Type × Condition", fontweight="bold")
        ax.set_ylabel("Query Type")
        ax.set_xlabel("Condition")

        plt.tight_layout()
        path = self.figures_dir / "query_type_heatmap.png"
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        logger.info("Saved → %s", path)
        return path

    def plot_agentic_metrics(self, df: pd.DataFrame) -> Path:
        """Plot agentic-specific metrics (link utilization, hop depth)."""
        agentic = df[df["retrieval_mode"] == "agentic"].copy()
        if agentic.empty:
            logger.warning("No agentic results to plot")
            return Path()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Link utilization
        sns.boxplot(data=agentic, x="condition", y="link_utilization", ax=ax1, palette="Set2")
        ax1.set_title("Link Utilization", fontweight="bold")
        ax1.set_ylabel("Fraction of Links Followed")

        # Hop depth
        sns.barplot(data=agentic, x="condition", y="max_hop_depth", ax=ax2, palette="Set2")
        ax2.set_title("Average Traversal Depth", fontweight="bold")
        ax2.set_ylabel("Hops")

        plt.tight_layout()
        path = self.figures_dir / "agentic_metrics.png"
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        logger.info("Saved → %s", path)
        return path

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------

    def generate_summary_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate LaTeX-ready summary table."""
        metrics = ["accuracy", "completeness", "grounding"]
        summary = df.groupby("condition")[metrics].agg(["mean", "std"]).round(3)
        summary.columns = [f"{m}_{s}" for m, s in summary.columns]

        # Save as LaTeX
        latex_path = self.tables_dir / "summary_table.tex"
        summary.to_latex(latex_path, float_format="%.3f")
        logger.info("Saved → %s", latex_path)

        # Save as CSV
        csv_path = self.tables_dir / "summary_table.csv"
        summary.to_csv(csv_path)

        return summary

    def generate_hypothesis_table(self, test_results: dict) -> pd.DataFrame:
        """Generate a table of hypothesis test results."""
        rows = []
        for hyp, metrics in test_results.items():
            for metric, test in metrics.items():
                rows.append({
                    "Hypothesis": hyp,
                    "Metric": metric,
                    "Mean Diff": test.get("mean_b", 0) - test.get("mean_a", 0),
                    "t-stat": test.get("t_stat", np.nan),
                    "p-value": test.get("p_value_corrected", np.nan),
                    "Cohen's d": test.get("cohens_d", np.nan),
                    "Significant": test.get("significant", False),
                    "n": test.get("n", 0),
                })

        table = pd.DataFrame(rows)
        latex_path = self.tables_dir / "hypothesis_tests.tex"
        table.to_latex(latex_path, float_format="%.4f", index=False)
        csv_path = self.tables_dir / "hypothesis_tests.csv"
        table.to_csv(csv_path, index=False)
        logger.info("Saved → %s", latex_path)
        return table

    # ------------------------------------------------------------------
    # Full analysis pipeline
    # ------------------------------------------------------------------

    def run_full_analysis(self) -> None:
        """Run the complete analysis pipeline."""
        results = self.load_results()
        if not results:
            return

        df = self.results_to_dataframe(results)
        logger.info("Analysis dataframe: %d rows", len(df))

        if df.empty:
            logger.error("No valid results to analyze")
            return

        # Summary statistics
        print("\n=== Summary Statistics ===")
        summary = self.generate_summary_table(df)
        print(summary.to_string())

        # Hypothesis tests
        print("\n=== Hypothesis Tests ===")
        tests = self.hypothesis_tests(df)
        test_table = self.generate_hypothesis_table(tests)
        print(test_table.to_string(index=False))

        # Visualizations
        self.plot_condition_comparison(df)
        self.plot_query_type_breakdown(df)
        self.plot_agentic_metrics(df)

        print(f"\nFigures → {self.figures_dir}")
        print(f"Tables  → {self.tables_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR / "raw")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    analyzer = ResultsAnalyzer(results_dir=args.results_dir, output_dir=args.output_dir)
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
