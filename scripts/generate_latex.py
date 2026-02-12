#!/usr/bin/env python3
"""Generate publication-quality LaTeX tables and figures from experiment results.

Reads all_results_*.json and produces:
  - tables/main_results.tex     : overall condition comparison
  - tables/stat_tests.tex       : hypothesis test results
  - tables/query_type.tex       : breakdown by query type
  - tables/domain.tex           : breakdown by domain
  - tables/agentic_metrics.tex  : agentic-specific metrics (C4-C6)
  - figures/condition_bars.pdf  : bar chart comparison
  - figures/heatmap.pdf         : accuracy × condition × query type heatmap
  - figures/domain_bars.pdf     : by-domain grouped bar chart
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

RESULTS_DIR = Path("results/raw")
OUT_TABLES = Path("paper/tables")
OUT_FIGURES = Path("paper/figures")

CONDITION_LABELS = {
    "C1": "Plain HTML, Std.",
    "C2": "HTML+JSON-LD, Std.",
    "C3": "Enhanced, Std.",
    "C4": "Plain HTML, Agent.",
    "C5": "HTML+JSON-LD, Agent.",
    "C6": "Enhanced, Agent.",
}

CONDITION_ORDER = ["C1", "C2", "C3", "C4", "C5", "C6"]
METRICS = ["accuracy", "completeness", "grounding"]
METRIC_LABELS = {"accuracy": "Accuracy", "completeness": "Completeness", "grounding": "Grounding"}

# Styling
sns.set_theme(style="whitegrid", font_scale=1.15)
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
})


def load_latest_results() -> dict:
    """Load the most recent combined results file."""
    files = sorted(RESULTS_DIR.glob("all_results_*.json"))
    if not files:
        print("ERROR: No results files found in", RESULTS_DIR)
        sys.exit(1)
    latest = files[-1]
    print(f"Loading: {latest}")
    with open(latest) as f:
        return json.load(f)


def results_to_df(results: dict) -> pd.DataFrame:
    """Flatten results to a pandas DataFrame."""
    rows = []
    for cond, entries in results.items():
        for r in entries:
            if "error" in r:
                continue
            m = r.get("metrics", {})
            links_followed = r.get("links_followed", [])
            links_available = r.get("links_available", [])
            agent_steps = r.get("agent_steps", [])
            rows.append({
                "condition": cond,
                "query_type": r.get("query_type", ""),
                "domain": r.get("domain", ""),
                "retrieval_mode": r.get("retrieval_mode", ""),
                "accuracy": m.get("accuracy_score", np.nan),
                "completeness": m.get("completeness_score", np.nan),
                "grounding": m.get("grounding_score", np.nan),
                "links_followed": len(links_followed) if isinstance(links_followed, list) else 0,
                "links_available": len(links_available) if isinstance(links_available, list) else 0,
                "max_hop_depth": r.get("max_hop_depth", 0),
                "num_tool_calls": len(agent_steps) if agent_steps else 0,
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# TABLE GENERATORS
# ─────────────────────────────────────────────────────────────────────────────
def gen_main_results_table(df: pd.DataFrame):
    """Table 2: Main results — mean (±std) for each metric per condition."""
    present = [c for c in CONDITION_ORDER if c in df["condition"].unique()]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Main results across experimental conditions (mean $\pm$ std).}",
        r"\label{tab:main_results}",
        r"\begin{tabular}{@{}clccc@{}}",
        r"\toprule",
        r"\textbf{ID} & \textbf{Condition} & \textbf{Accuracy} & \textbf{Completeness} & \textbf{Grounding} \\",
        r"\midrule",
    ]
    for cond in present:
        sub = df[df["condition"] == cond]
        label = CONDITION_LABELS.get(cond, cond)
        acc = f"{sub['accuracy'].mean():.2f} $\\pm$ {sub['accuracy'].std():.2f}"
        comp = f"{sub['completeness'].mean():.2f} $\\pm$ {sub['completeness'].std():.2f}"
        grd_mean = sub['grounding'].mean()
        grd = f"{grd_mean:.2f} $\\pm$ {sub['grounding'].std():.2f}" if not np.isnan(grd_mean) else "---"
        lines.append(f"{cond} & {label} & {acc} & {comp} & {grd} \\\\")
        if cond in ("C3",):  # separator between standard and agentic
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    text = "\n".join(lines)
    (OUT_TABLES / "main_results.tex").write_text(text)
    print("  ✓ tables/main_results.tex")


def gen_stat_tests_table(df: pd.DataFrame):
    """Table 3: Hypothesis test results."""
    comparisons = [
        ("H1", "C1", "C2", "Structured data (standard RAG)"),
        ("H1′", "C1", "C3", "Enhanced pages (standard RAG)"),
        ("H2", "C2", "C5", "Agentic RAG vs standard"),
        ("H3", "C5", "C6", "Enhanced pages (agentic RAG)"),
        ("Full", "C1", "C6", "Full pipeline vs baseline"),
    ]
    present_conds = set(df["condition"].unique())
    n_tests = sum(1 for _, a, b, _ in comparisons if a in present_conds and b in present_conds) * 3

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Paired $t$-tests with Bonferroni correction ($\alpha = 0.05$, "
        + f"$n_{{\\text{{tests}}}} = {n_tests}$"
        + r").}",
        r"\label{tab:stat_tests}",
        r"\begin{tabular}{@{}llrrrrl@{}}",
        r"\toprule",
        r"\textbf{Hyp.} & \textbf{Metric} & \textbf{$\Delta$} & "
        r"\textbf{$t$} & \textbf{$p_{\text{adj}}$} & \textbf{$d$} & \textbf{Sig.} \\",
        r"\midrule",
    ]

    for hyp_label, cond_a, cond_b, desc in comparisons:
        if cond_a not in present_conds or cond_b not in present_conds:
            continue
        lines.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{desc} ({cond_a} vs {cond_b})}}}} \\\\")
        for metric in METRICS:
            a = df[df["condition"] == cond_a][metric].dropna().values
            b = df[df["condition"] == cond_b][metric].dropna().values
            n = min(len(a), len(b))
            if n < 2:
                continue
            t_stat, p_val = stats.ttest_rel(a[:n], b[:n])
            p_adj = min(p_val * n_tests, 1.0)
            d = (np.mean(b[:n]) - np.mean(a[:n])) / np.std(a[:n]) if np.std(a[:n]) > 0 else 0
            delta = np.mean(b[:n]) - np.mean(a[:n])
            sig = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else "n.s."
            lines.append(
                f"  & {METRIC_LABELS[metric]} & {delta:+.2f} & "
                f"{t_stat:.2f} & {p_adj:.1e} & {d:.2f} & {sig} \\\\"
            )
        lines.append(r"\midrule")

    # Remove trailing \midrule
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]
    text = "\n".join(lines)
    (OUT_TABLES / "stat_tests.tex").write_text(text)
    print("  ✓ tables/stat_tests.tex")


def gen_query_type_table(df: pd.DataFrame):
    """Table 4: Accuracy by condition × query type."""
    present = [c for c in CONDITION_ORDER if c in df["condition"].unique()]
    qtypes = ["factual", "relational", "comparative"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Mean accuracy by condition and query type.}",
        r"\label{tab:query_type}",
        r"\begin{tabular}{@{}l" + "c" * len(present) + r"@{}}",
        r"\toprule",
        r"\textbf{Query Type} & " + " & ".join(f"\\textbf{{{c}}}" for c in present) + r" \\",
        r"\midrule",
    ]
    for qt in qtypes:
        vals = []
        for c in present:
            sub = df[(df["condition"] == c) & (df["query_type"] == qt)]
            vals.append(f"{sub['accuracy'].mean():.2f}" if len(sub) > 0 else "---")
        lines.append(f"{qt.capitalize()} & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    text = "\n".join(lines)
    (OUT_TABLES / "query_type.tex").write_text(text)
    print("  ✓ tables/query_type.tex")


def gen_domain_table(df: pd.DataFrame):
    """Table 5: Accuracy by condition × domain."""
    present = [c for c in CONDITION_ORDER if c in df["condition"].unique()]
    domains = sorted(df["domain"].unique())
    domain_labels = {
        "wordlift_blog": "WordLift Blog",
        "salzburgerland": "SalzburgerLand",
        "blackbriar": "BlackBriar",
    }

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Mean accuracy by condition and domain.}",
        r"\label{tab:domain}",
        r"\begin{tabular}{@{}l" + "c" * len(present) + r"@{}}",
        r"\toprule",
        r"\textbf{Domain} & " + " & ".join(f"\\textbf{{{c}}}" for c in present) + r" \\",
        r"\midrule",
    ]
    for dom in domains:
        vals = []
        for c in present:
            sub = df[(df["condition"] == c) & (df["domain"] == dom)]
            vals.append(f"{sub['accuracy'].mean():.2f}" if len(sub) > 0 else "---")
        label = domain_labels.get(dom, dom)
        lines.append(f"{label} & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    text = "\n".join(lines)
    (OUT_TABLES / "domain.tex").write_text(text)
    print("  ✓ tables/domain.tex")


def gen_agentic_table(df: pd.DataFrame):
    """Table 6: Agentic-specific metrics for C4-C6."""
    agentic = df[df["retrieval_mode"] == "agentic"]
    if agentic.empty:
        print("  ⏭ Skipping agentic table (no agentic results yet)")
        return

    present = [c for c in ["C4", "C5", "C6"] if c in agentic["condition"].unique()]
    if not present:
        return

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Agentic-specific metrics across conditions.}",
        r"\label{tab:agentic}",
        r"\begin{tabular}{@{}lccc@{}}",
        r"\toprule",
        r"\textbf{Metric} & " + " & ".join(f"\\textbf{{{c}}}" for c in present) + r" \\",
        r"\midrule",
    ]
    for metric, label in [
        ("links_followed", "Links followed"),
        ("links_available", "Links available"),
        ("max_hop_depth", "Max hop depth"),
        ("num_tool_calls", "Tool calls"),
    ]:
        vals = []
        for c in present:
            sub = agentic[agentic["condition"] == c]
            vals.append(f"{sub[metric].mean():.1f}")
        lines.append(f"{label} & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    text = "\n".join(lines)
    (OUT_TABLES / "agentic_metrics.tex").write_text(text)
    print("  ✓ tables/agentic_metrics.tex")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE GENERATORS
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    "C1": "#8B9DC3",  # muted blue
    "C2": "#6B8DD6",  # blue
    "C3": "#3A6BC5",  # rich blue
    "C4": "#E8A87C",  # muted orange
    "C5": "#E07B39",  # orange
    "C6": "#D35400",  # deep orange
}


def fig_condition_bars(df: pd.DataFrame):
    """Bar chart: accuracy + completeness for each condition."""
    present = [c for c in CONDITION_ORDER if c in df["condition"].unique()]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    for ax, metric in zip(axes, ["accuracy", "completeness"]):
        means = [df[df["condition"] == c][metric].mean() for c in present]
        stds = [df[df["condition"] == c][metric].std() for c in present]
        colors = [COLORS.get(c, "#999") for c in present]

        bars = ax.bar(present, means, yerr=stds, capsize=3, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_title(METRIC_LABELS[metric], fontsize=13, fontweight="bold")
        ax.set_ylim(0, 5.5)
        ax.set_ylabel("Score (1–5)" if metric == "accuracy" else "")
        ax.axhline(y=df[df["condition"] == "C1"][metric].mean(), color="#999", ls="--", lw=0.8, alpha=0.6)

        # Value labels
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                    f"{mean:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig.savefig(OUT_FIGURES / "condition_bars.pdf")
    fig.savefig(OUT_FIGURES / "condition_bars.png")
    plt.close(fig)
    print("  ✓ figures/condition_bars.pdf")


def fig_heatmap(df: pd.DataFrame):
    """Heatmap: accuracy by condition × query type."""
    present = [c for c in CONDITION_ORDER if c in df["condition"].unique()]
    qtypes = ["factual", "relational", "comparative"]

    pivot = df.pivot_table(values="accuracy", index="query_type", columns="condition",
                           aggfunc="mean").reindex(index=qtypes, columns=present)

    fig, ax = plt.subplots(figsize=(max(5, len(present) * 1.2), 3.5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", vmin=1, vmax=5,
                linewidths=0.5, ax=ax, cbar_kws={"label": "Accuracy"})
    ax.set_title("Accuracy by Condition × Query Type", fontsize=12, fontweight="bold")
    ax.set_ylabel("")
    ax.set_xlabel("")
    plt.tight_layout()
    fig.savefig(OUT_FIGURES / "heatmap.pdf")
    fig.savefig(OUT_FIGURES / "heatmap.png")
    plt.close(fig)
    print("  ✓ figures/heatmap.pdf")


def fig_domain_bars(df: pd.DataFrame):
    """Grouped bar chart: accuracy by domain for each condition."""
    present = [c for c in CONDITION_ORDER if c in df["condition"].unique()]
    domains = sorted(df["domain"].unique())
    domain_labels = {
        "wordlift_blog": "WordLift\nBlog",
        "salzburgerland": "Salzburger-\nLand",
        "blackbriar": "Black-\nBriar",
    }

    n_conds = len(present)
    n_doms = len(domains)
    x = np.arange(n_doms)
    width = 0.8 / n_conds

    fig, ax = plt.subplots(figsize=(max(6, n_doms * 2.5), 4.5))
    for i, cond in enumerate(present):
        vals = [df[(df["condition"] == cond) & (df["domain"] == d)]["accuracy"].mean() for d in domains]
        ax.bar(x + i * width - (n_conds - 1) * width / 2, vals, width,
               label=cond, color=COLORS.get(cond, "#999"), edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([domain_labels.get(d, d) for d in domains])
    ax.set_ylabel("Accuracy (1–5)")
    ax.set_ylim(0, 5.5)
    ax.set_title("Accuracy by Domain", fontsize=12, fontweight="bold")
    ax.legend(title="Condition", ncol=min(n_conds, 3), fontsize=8, title_fontsize=9)
    plt.tight_layout()
    fig.savefig(OUT_FIGURES / "domain_bars.pdf")
    fig.savefig(OUT_FIGURES / "domain_bars.png")
    plt.close(fig)
    print("  ✓ figures/domain_bars.pdf")


def fig_improvement_waterfall(df: pd.DataFrame):
    """Waterfall chart showing incremental improvement from C1 to C6."""
    present = [c for c in CONDITION_ORDER if c in df["condition"].unique()]
    if len(present) < 2:
        return

    baseline = df[df["condition"] == present[0]]["accuracy"].mean()
    fig, ax = plt.subplots(figsize=(8, 4.5))

    cumulative = baseline
    for i, cond in enumerate(present):
        val = df[df["condition"] == cond]["accuracy"].mean()
        color = COLORS.get(cond, "#999")
        ax.bar(i, val, color=color, edgecolor="white", linewidth=0.5)
        ax.text(i, val + 0.1, f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        if i > 0:
            delta = val - df[df["condition"] == present[i-1]]["accuracy"].mean()
            sign = "+" if delta >= 0 else ""
            ax.text(i, val - 0.3, f"({sign}{delta:.2f})", ha="center", va="top", fontsize=8, color="#555")

    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([f"{c}\n{CONDITION_LABELS.get(c, c)}" for c in present], fontsize=8)
    ax.set_ylabel("Mean Accuracy (1–5)")
    ax.set_ylim(0, 5.5)
    ax.set_title("Accuracy Progression Across Conditions", fontsize=12, fontweight="bold")

    # Baseline reference line
    ax.axhline(y=baseline, color="#999", ls="--", lw=0.8, alpha=0.6, label=f"Baseline: {baseline:.2f}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(OUT_FIGURES / "improvement_waterfall.pdf")
    fig.savefig(OUT_FIGURES / "improvement_waterfall.png")
    plt.close(fig)
    print("  ✓ figures/improvement_waterfall.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)

    results = load_latest_results()
    df = results_to_df(results)
    print(f"DataFrame: {len(df)} rows, conditions: {sorted(df['condition'].unique())}")
    print()

    print("Generating LaTeX tables...")
    gen_main_results_table(df)
    gen_stat_tests_table(df)
    gen_query_type_table(df)
    gen_domain_table(df)
    gen_agentic_table(df)
    print()

    print("Generating figures...")
    fig_condition_bars(df)
    fig_heatmap(df)
    fig_domain_bars(df)
    fig_improvement_waterfall(df)
    print()

    print("Done! Files in:")
    print(f"  Tables:  {OUT_TABLES.resolve()}")
    print(f"  Figures: {OUT_FIGURES.resolve()}")


if __name__ == "__main__":
    main()
