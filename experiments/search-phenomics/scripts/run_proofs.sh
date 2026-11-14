#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
command -v lean >/dev/null || { echo 'Lean compiler missing'; exit 2; }
echo "Lean version: $(lean --version)"
printf 'Compiling and proving all %s source files\n' "$(find lean -name '*.lean' | wc -l)"
for f in lean/*.lean; do
  echo "[PROVE] ${f}"
  lean "$f"
done
# Negative controls: attempt two false proofs using the *same* declarations.
for case in fabricated_measurement fabricated_causality; do
  neg="$(mktemp --suffix=.lean)"
  log="$(mktemp)"
  cat lean/SearchPhenomics.lean > "$neg"
  if [ "$case" = fabricated_measurement ]; then
    printf '\nnamespace SearchPhenomics\nexample : canAdmitAsMeasurement .agentSummary = true := by rfl\nend SearchPhenomics\n' >> "$neg"
  else
    printf '\nnamespace SearchPhenomics\nexample : claimCanBeConcludedFromObservations .googleSignalCausality = true := by rfl\nend SearchPhenomics\n' >> "$neg"
  fi
  if lean "$neg" > "$log" 2>&1; then
    echo "[FAIL] Lean unexpectedly accepted forbidden claim: $case"
    cat "$log"; rm -f "$neg" "$log"; exit 1
  fi
  if ! grep -Eq "(tactic 'rfl' failed|type mismatch|unsolved goals|rfl failed)" "$log"; then
    echo "[FAIL] Rejected, but without expected proof-failure diagnostic: $case"
    cat "$log"; rm -f "$neg" "$log"; exit 1
  fi
  echo "[EXPECTED REJECTION] $case"
  grep -m 2 -E "(tactic 'rfl' failed|type mismatch|unsolved goals|rfl failed)" "$log" || true
  rm -f "$neg" "$log"
done
printf 'PASS: all policy theorems compiled; both false proofs rejected.\n'
