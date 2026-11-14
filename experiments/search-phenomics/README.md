# Search Phenomics — Experiment 01: Formal Validation

This is a standalone Lean 4 proof harness for the Search Phenomics / SEOntology research pipeline, isolated from the existing SEO 3.0 paper and production WordLift Agent.

The theorems formalize inference gates, **not** Google's undocumented ranking weights or empirical truth of an AI answer. Without archived raw provider evidence, independent site inventories, matched intent and controls, the current topical-authority case must not be promoted to a validated association.

Run locally (Lean 4.23.0 required): `bash scripts/run_proofs.sh`.

The GitHub Actions workflow compiles four proof modules, executes `#guard` and `#eval` tests, and attempts **two false lemmas** that Lean must reject with a proof-specific diagnostic.

Expected trace:

```text
Actual Agent-summary pilot may review topical association: false
Synthetic fully measured scenario may review association: true
Observations can establish Google's causal ranking mechanism: false
[EXPECTED REJECTION] fabricated_measurement
[EXPECTED REJECTION] fabricated_causality
```

Passing means **logical policy verification** only. It does not validate source provenance, causal ranking relationships, or independent competitor graph coverage. The empirical study retains separate NO-GO gates for raw provider snapshots, independent site-topic graphs and preregistered statistical tests.

The suite uses Lean core only (no Mathlib). It does not modify SEOntology core terms or the API gateway.
