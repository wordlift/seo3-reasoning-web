/-!
Search Phenomics: signal declarations do not entail deployed Google ranking use.
These are formal statements about evidence policy, not empirical proof.
-/
namespace ProofCarryingSEO

structure SignalState where
  declaredInSchema : Bool
  usedInRanking : Bool

theorem declaration_does_not_entail_use :
    ¬ (∀ s : SignalState,
       s.declaredInSchema = true → s.usedInRanking = true) := by
  intro assumption
  have impossible : false = true := assumption ⟨true, false⟩ rfl
  cases impossible

inductive EvidenceKind
  | schemaDeclaration
  | serpObservation
  | controlledIntervention
  deriving DecidableEq, Repr

def eligibleForCausalReview : EvidenceKind → Bool
  | .schemaDeclaration => false
  | .serpObservation => false
  | .controlledIntervention => true

theorem schema_only_ineligible :
    eligibleForCausalReview .schemaDeclaration = false := rfl

theorem observation_only_ineligible :
    eligibleForCausalReview .serpObservation = false := rfl

theorem causal_review_requires_intervention
    (e : EvidenceKind) (h : eligibleForCausalReview e = true) :
    e = .controlledIntervention := by
  cases e <;> simp [eligibleForCausalReview] at h ⊢

structure SerpCapture where
  hasDestinationUrls : Bool
  completeTopTen : Bool
  timestampAndLocalePresent : Bool

def admissibleForDomainAnalysis (s : SerpCapture) : Bool :=
  s.hasDestinationUrls && s.completeTopTen && s.timestampAndLocalePresent

theorem redirects_are_not_domain_evidence (s : SerpCapture)
    (h : s.hasDestinationUrls = false) :
    admissibleForDomainAnalysis s = false := by
  simp [admissibleForDomainAnalysis, h]

end ProofCarryingSEO
