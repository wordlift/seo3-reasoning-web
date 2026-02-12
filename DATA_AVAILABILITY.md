# Data Availability

## What is included in this repository

This repository contains all **code, templates, configuration, and paper materials**
needed to understand and replicate the methodology described in the paper
*"Structured Linked Data in Agentic RAG"*.

Specifically, the repository includes:

- **Source code** (`src/`): Data collection, document transformation, indexing,
  retrieval (standard + agentic RAG), evaluation metrics, and analysis scripts.
- **Configuration** (`config/`): Experiment parameters, domain definitions,
  and GCP setup scripts.
- **Templates** (`templates/`): The enhanced entity page HTML template and
  `llms.txt`-style agent instruction template.
- **Paper** (`paper/`): Full LaTeX source, tables, figures, and references.
- **Scripts** (`scripts/`): Utility scripts for data fetching, LaTeX generation,
  and judge failure repair.

## What is NOT included

The following data is excluded from the public repository to protect
client confidentiality:

| Directory | Content | Reason |
|-----------|---------|--------|
| `data/raw/` | Raw HTML pages from partner websites | Client content |
| `data/processed/` | Processed entity pages (3 formats × 217 entities) | Client content |
| `data/queries/` | 305 evaluation queries with ground truth | Derivative of client data |
| `results/raw/` | Full evaluation results (1,785 query-answer pairs) | Contains client data in LLM answers |

## How to obtain the data

Researchers wishing to access the experimental data for replication purposes
may contact the authors:

- **Andrea Volpini** — andrea@wordlift.io

Data will be shared under a research-use agreement that respects the
confidentiality of the partner organizations (BlackBriar, SalzburgerLand,
Express Legal Funding, and WordLift Blog).

## Reproducing with your own data

The experiment can be fully reproduced with any set of websites that have
Schema.org JSON-LD markup. To run with your own data:

1. Define your domains in `data/domains.json`
2. Set up API keys in `.env` (see `.env.example`)
3. Run `python scripts/fetch_web_pages.py` to collect raw HTML
4. Run the document transformer to generate the 3 format variants
5. Set up Vertex AI Vector Search 2.0 collections
6. Run the experiment with `python -m src.evaluation.runner`

See the [README](README.md) for detailed instructions.
