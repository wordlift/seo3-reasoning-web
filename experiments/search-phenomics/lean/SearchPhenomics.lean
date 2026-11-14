/-!
Search Phenomics Experiment 01 — verifiable evidence-policy obligations.
Formal validity never certifies the empirical premises, Google infrastructure,
ranking weight, or causal mechanism.
-/
namespace SearchPhenomics

inductive Source
  | rawProvider
  | agentSummary
  | schemaReport
  deriving Repr, DecidableEq

def canAdmitAsMeasurement : Source → Bool
  | .rawProvider => true
  | .agentSummary => false
  | .schemaReport => false

theorem agent_summary_not_measurement :
    canAdmitAsMeasurement .agentSummary = false := rfl

theorem reported_schema_not_measurement :
    canAdmitAsMeasurement .schemaReport = false := rfl

structure StudyEvidence where
  rawCapture : Bool
  resolvedUrls : Bool
  independentSiteProfiles : Bool
  matchedIntent : Bool
  confoundersAssessed : Bool

def canReviewAssociation (e : StudyEvidence) : Bool :=
  e.rawCapture && e.resolvedUrls && e.independentSiteProfiles &&
  e.matchedIntent && e.confoundersAssessed

theorem no_raw_capture_no_association (e : StudyEvidence)
    (h : e.rawCapture = false) :
    canReviewAssociation e = false := by
  simp [canReviewAssociation, h]

theorem no_independent_site_graph_no_association (e : StudyEvidence)
    (h : e.independentSiteProfiles = false) :
    canReviewAssociation e = false := by
  simp [canReviewAssociation, h, Bool.and_assoc]

inductive Claim
  | schemaReported
  | observedAssociation
  | googleSignalCausality
  deriving Repr, DecidableEq

def claimCanBeConcludedFromObservations : Claim → Bool
  | .schemaReported => true
  | .observedAssociation => true
  | .googleSignalCausality => false

theorem observational_evidence_not_google_mechanism :
    claimCanBeConcludedFromObservations .googleSignalCausality = false := rfl

end SearchPhenomics

/-! Executable assay checks: #guard is checked by the Lean compiler. -/
namespace SearchPhenomics

private def instrumentPilot : StudyEvidence := {
  rawCapture := false
  resolvedUrls := true
  independentSiteProfiles := false
  matchedIntent := true
  confoundersAssessed := false
}

private def syntheticFullyMeasured : StudyEvidence := {
  rawCapture := true
  resolvedUrls := true
  independentSiteProfiles := true
  matchedIntent := true
  confoundersAssessed := true
}

#guard canAdmitAsMeasurement .rawProvider
#guard !(canAdmitAsMeasurement .agentSummary)
#guard !(canAdmitAsMeasurement .schemaReport)
#guard !(canReviewAssociation instrumentPilot)
#guard canReviewAssociation syntheticFullyMeasured
#guard !(claimCanBeConcludedFromObservations .googleSignalCausality)

#eval "Actual Agent-summary pilot may review topical association: " ++ toString (canReviewAssociation instrumentPilot)
#eval "Synthetic fully measured scenario may review association: " ++ toString (canReviewAssociation syntheticFullyMeasured)
#eval "Observations can establish Google's causal ranking mechanism: " ++ toString (claimCanBeConcludedFromObservations .googleSignalCausality)

end SearchPhenomics
