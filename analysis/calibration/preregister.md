---
pre-registration-commit: <TBD -- filled at commit time by Abe>
date-registered: 2026-04-22
venue: analysis/calibration/preregister.md (this file), immutable after commit
author: o-cotformer-skel-1 on behalf of Abe (ab3u21)
scope: Protocol D-calibration four-gate validation ladder plus Pilot 1 reproduction gate
---

# Pre-registration -- Protocol D-calibration and Pilot 1 gate

This document pre-registers the frozen thresholds for the Protocol
D-calibration four-gate validation ladder (`docs/extend-notes.md`
§1.3 "D-cal four-gate validation ladder") and the Pilot 1 counting
reproduction gate (`docs/extend-notes.md` §1.2 RQ9 "Twin-pilot
design" and §1.8 "Empirical step-time validation gate"). Every
threshold below is immutable after the skeleton commit; any post-hoc
adjustment requires a dated amendment in `docs/extend-notes.md` §1.9
M2 Decision Log AND an explicit reference in the commit message to
the amending DEC-M2-NNN identifier.

## Section 1 -- Protocol D-calibration four-gate thresholds

The four gates are evaluated in order; failure at any gate triggers
the fallback in the corresponding row rather than continuation.

### Gate 1 -- Baseline gate

- **Test**: Tier-1 model's Condition-A attention-target accuracy.
- **PASS threshold**: accuracy >= 0.5 on the pooled Condition-A set
  at the Tier-1 substrate (GPT-2-large).
- **Rationale**: ensures the Tier-1 model has functional induction
  heads before the calibration ladder is run. A model that cannot
  solve the induction task cannot be used to validate that low
  entropy implies target-accurate attention.
- **Fallback on fail**: Tier-2 substrate (ADM C5) replaces Tier-1 as
  the primary calibration anchor and the ladder restarts from Gate 2.

### Gate 2 -- Sink-correction monotonicity

- **Test**: absolute difference between full and no-sink Shannon
  entropy of the attention distribution, |H_full - H_no_sink|, per
  sequence.
- **PASS threshold**: |H_full - H_no_sink| < 1 bit on both Condition
  A and Condition B (pooled across sequences within each condition).
- **Rationale**: attention-sink-induced entropy depression (Xiao et
  al. [26], Gu et al. [46]) must be corrected before the calibration
  monotonicity claim can be evaluated. A > 1 bit shift indicates the
  sink dominates the entropy signal.
- **Fallback on fail**: `H_no_sink` becomes the operative metric
  throughout and a separate sink-mass-vs-survival-fraction
  correlation is reported alongside the calibration verdict.

### Gate 3 -- Triangulation gate

- **Test**: Tier-1 (GPT-2-large) and Tier-2 (ADM C5) calibration
  verdicts are compared on the PASS / AMBIGUOUS / FAIL axis.
- **PASS threshold**: Tier-1 and Tier-2 verdicts agree.
- **Fallback on fail**: Tier-3 (24L standard transformer; contingency
  only, ~1-1.5 L4-days fresh training) is triggered; if Tier-3 is
  unavailable within the budget window, Tier-2 (ADM C5) governs the
  final verdict because the calibration must validate the metric on
  the architecture where DV-2 will be applied at test time.

### Gate 4 -- Spearman and classifier gate

- **Test 1 (Spearman)**: rank correlation of per-sequence
  (entropy, attention_target_accuracy) pooled across 2000 sequences
  (1000 per condition at the primary ladder).
- **Test 2 (classifier)**: linear classifier on
  (entropy, attention_target_accuracy) separating Condition A from
  Condition B.
- **PASS threshold**: Spearman rho <= -0.5 AND classifier accuracy
  >= 0.8.
- **Rationale**: at N = 2000, statistical power to detect rho = -0.5
  is greater than 0.999 at alpha = 0.001 (Cohen 1988 [49]); the
  combined criterion requires BOTH a strong monotone relationship
  AND a separability guarantee. Monotonicity alone is insufficient
  because a monotone but unreliable mapping fails to distinguish
  Condition A (low entropy, high accuracy) from a hypothetical
  Condition B' (low entropy, low accuracy).
- **Verdict table**:
  - **PASS** (Spearman rho <= -0.5 AND classifier >= 0.8): monotone
    interpretation empirically defensible on the calibrated
    substrate. DV-2 is operationalised as the calibrated isolation
    classifier; classifier-accuracy gate is re-checked at test time.
  - **AMBIGUOUS** (Spearman rho in (-0.5, -0.3)): rerun with
    N = 4000 per condition to resolve. If ambiguity persists, DV-2
    is downgraded to single-measurement-preliminary per the §1.9
    scope-shedding DEC-M2-030 trigger.
  - **FAIL** (Spearman rho > -0.3 OR linear classifier on entropy
    alone >= 0.85): monotone interpretation refuted. Apply the
    REPLACE-WITH-TOP-K-MASS fallback: DV-2 is replaced by "fraction
    of attention mass at repeat 5 landing on context tokens that
    survived to the same repeat or deeper".

## Section 2 -- Pilot 1 reproduction gate (counting)

The Pilot 1 arm (Chang and Bisk 4L standard Transformer at
n_embd=1024, Chang and Bisk-exact regime per DEC-M2-019 and the
seven-divergence reconciliation DEC-M2-020) carries two hard gates
that the main sweep cannot proceed without.

### Gate P1-A -- IND accuracy gate

- **Test**: per-position OOD accuracy at length 50 (the train-side
  OOD boundary for the te200 variant) at Pilot 1 completion.
- **PASS threshold**: IND accuracy >= 80 per cent.
- **Rationale**: Chang and Bisk [25] report the 4L standard
  transformer reaches near-perfect accuracy on length 50 under this
  regime; failing to reproduce that baseline invalidates the
  reference point for every cross-architecture comparison in RQ9.
- **Fallback on fail**: RQ9 sub-stream sheds to Milestone 3 per the
  §0 scope-shedding trigger 1. The M2 analytic sub-stream proceeds
  unaffected.

### Gate P1-B -- Per-step wall time gate

- **Test**: per-step wall time of Pilot 1's training loop on the
  project's 2x L4 node, averaged over the steady-state region (the
  first 10 per cent of steps are excluded as warm-up).
- **PASS threshold**: per-step time within 50 per cent of the
  projected value recorded in `docs/extend-notes.md` §1.8 "Empirical
  step-time validation gate".
- **Rationale**: projection error greater than 50 per cent cascades
  into wall-clock overruns on the 42-run main sweep; a deviation at
  this scale is a signal that the queue model or the data-loader is
  misconfigured.
- **Fallback on fail**: budget revision before the main sweep is
  submitted; triggers a review with Abe and a revised sequencing in
  the §1.8 Compute Budget table.

## Section 3 -- Audit trail

Any change to a threshold in Section 1 or Section 2 after the
skeleton commit requires (a) a dated DEC-M2-NNN entry in
`docs/extend-notes.md` §1.9 describing the change and its rationale,
(b) a reference to that identifier in the amending commit message,
and (c) a cross-reference line appended to the appropriate gate
section above. The file header's ``pre-registration-commit`` frontmatter
is the authoritative anchor for the skeleton-commit SHA; it is set
once by Abe at commit time and is never rewritten.
