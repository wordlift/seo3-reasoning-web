# SEO 3.0: The Reasoning Web

**Structured Linked Data for Agentic RAG — An Empirical Study**

This repository contains the code, experiment infrastructure, and paper source
for our study investigating how structured linked data (Schema.org markup and
knowledge graph entity pages served by a Linked Data Platform) impacts retrieval
accuracy in RAG systems built on Vertex AI Vector Search 2.0 and the Google
Agent Development Kit (ADK).

## Key Findings

| Hypothesis | Result | Effect |
|-----------|--------|--------|
| **H1**: JSON-LD alone improves RAG accuracy | ❌ Not significant (p=1.0) | Δ = +0.07 |
| **H2**: Agentic RAG outperforms standard RAG | ✅ Significant (p=0.001) | +14.4%, d=0.27 |
| **H3**: Enhanced entity pages improve accuracy | ✅ Significant (p<1e-11) | +29.5–30.8%, d=0.55–0.60 |

**Why H1 fails**: Our pipeline ingests pages as flat text truncated at 20k
characters. 82% of documents exceed this limit, and the JSON-LD block sits
right at the truncation boundary (median: char 18,510). Production search
engines like Google extract JSON-LD *separately* — a fundamentally different
architecture. See the paper for details.

## The SEO 3.0 Framework

We propose three eras of search optimization:

- **SEO 1.0 — Document Ranking** (1998–2011): Keywords and links
- **SEO 2.0 — Structured Data** (2011–2023): Schema.org, knowledge panels
- **SEO 3.0 — The Reasoning Web** (2023–present): AI systems that reason and act

And three tiers of AI visibility:

1. **Citations** — Is your content retrieved and attributed?
2. **Reasoning** — Can the AI reason correctly over your content?
3. **Actions** — Can the AI agent act on your content?

## Domains Under Study

| Domain | Vertical | Entity Types |
|--------|----------|-------------|
| [BlackBriar](https://myblackbriar.com/) | Advisors | Services, team members, insights |
| [SalzburgerLand](https://www.salzburgerland.com/) | Travel / Tourism | Places, attractions, cards |
| [Express Legal Funding](https://www.expresslegalfunding.com/) | Legal / Finance | Services, processes, state guides |
| [WordLift Blog](https://wordlift.io/blog/) | Editorial | Articles, concepts (Knowledge Graph, SEO, NER) |

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Configure GCP credentials
gcloud auth application-default login

# Collect entities from the Linked Data Platform
python -m src.dataset.collector --config config/experiment_config.yaml

# Generate document variants (plain HTML, HTML+JSON-LD, enhanced)
python -m src.dataset.transformer --config config/experiment_config.yaml

# Generate test queries with ground truth
python -m src.dataset.query_generator --config config/experiment_config.yaml

# Set up Vertex AI Vector Search 2.0 collections
python -m src.indexing.vectorsearch --setup --ingest all

# Run experiments (all 6 conditions)
python -m src.evaluation.runner --config config/experiment_config.yaml

# Generate analysis, figures, and LaTeX tables
python -m src.evaluation.analysis --results-dir results/raw/ --output-dir results/
```

## Project Structure

```
├── config/                 # Configuration files
├── data/                   # Dataset (raw, processed, queries) — see DATA_AVAILABILITY.md
├── src/                    # Source code
│   ├── dataset/            # Data collection & curation
│   ├── indexing/           # Vertex AI Vector Search 2.0
│   ├── retrieval/          # Standard & Agentic RAG pipelines
│   └── evaluation/         # Metrics, runner, analysis
├── templates/              # Enhanced entity page & llms.txt templates
├── paper/                  # LaTeX paper source (LNCS format)
├── scripts/                # Utility scripts
└── results/                # Experiment outputs (gitignored)
```

## Requirements

- Python 3.11+
- Google Cloud project with Vertex AI APIs enabled
- `gcloud` CLI authenticated

## Data Availability

The experimental data (raw HTML, processed documents, evaluation results)
is excluded from this repository to protect client confidentiality.
See [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) for details on how to
request data for replication or reproduce the experiment with your own websites.

## License

- **Code**: [MIT License](LICENSE)
- **Paper & Figures**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{volpini2026seo3,
  title     = {Structured Linked Data in Agentic {RAG}: From {SEO} to the Reasoning Web},
  author    = {Volpini, Andrea},
  booktitle = {Proceedings of the International Semantic Web Conference (ISWC)},
  year      = {2026}
}
```
