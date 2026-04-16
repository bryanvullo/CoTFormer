# Extension Notes

> Scientific investigation and architectural extension of the CoTFormer
> (Mohtashami et al., ICLR 2025). This document records the team's
> methodology, research questions, experimental designs, and extension
> proposals across three milestones.
>
> For reproduction divergences, see `docs/reprod-notes.md`.
> For training commands, see `experiments.md`.

## Table of Contents

- [0. Methodology](#0-methodology)
- [1. Milestone 2: Mechanistic Analysis](#1-milestone-2-mechanistic-analysis)
  - [1.1 Ontological Framework](#11-ontological-framework)
  - [1.2 Research Questions](#12-research-questions)
  - [1.3 Experimental Protocols](#13-experimental-protocols)
  - [1.4 Theoretical Predictions](#14-theoretical-predictions)
  - [1.5 Analysis Matrix](#15-analysis-matrix)
  - [1.6 Reliability Discipline and Triangulation](#16-reliability-discipline-and-triangulation)
  - [1.7 Code and Infrastructure Strategy](#17-code-and-infrastructure-strategy)
  - [1.8 Compute Budget and Sequencing](#18-compute-budget-and-sequencing)
  - [1.9 M2 Decision Log](#19-m2-decision-log)
- [2. Milestone 3: Architectural Extensions](#2-milestone-3-architectural-extensions)
  - [2.1 Multi-Head Latent Attention (MLA)](#21-multi-head-latent-attention-mla)
  - [2.2 Manifold-Constrained HyperConnections (mcHC)](#22-manifold-constrained-hyperconnections-mchc)
  - [2.3 Causal Cross-Repeat Mask for ADM](#23-causal-cross-repeat-mask-for-adm)
  - [2.4 Residual Attention](#24-residual-attention)
- [Appendix A: MLA Dimensionality Reference](#appendix-a-mla-dimensionality-reference)
- [References](#references)

---

## 0. Methodology

### Three Milestones

| # | Milestone | Goal | Status |
|---|-----------|------|--------|
| 1 | Reproduce | Verify paper claims (Tables 1-2, Figures 2-5, Section 5) | Near-complete |
| 2 | Unpick and Analyse | Understand what iterative computation computes | Active |
| 3 | Extend | Improve architecture informed by Milestone 2 findings | Blocked on M2 |

The milestones are sequential: each informs the next. Milestone 2 is
self-contained -- understanding is the goal, not a stepping stone to
engineering. The fact that we use M2 insights to design M3 extensions
is a consequence of the scientific method, not the purpose of M2.

M2 has two sub-streams:

- **M2 main** (analytic): 10 protocols plus synthesis on frozen C1-C5
  checkpoints. Eval-only. Compute: ~2.3 h GPU + ~5.5 h CPU. Single
  HPC envelope split into a GPU job and a CPU job.
- **M2 counting** ([RQ9](#rq9-recurrent-inductive-bias-for-algorithmic-counting)):
  fresh training of two baseline architectures and a 1L-4R CoTFormer
  across a width sweep of n_embd in {128, 256, 512, 1024}. ~4-5
  L4-days total. Sequential queue after M2 main; Aras Kavuncu
  (tak1e25) is the human reviewer.

### Scientific Ethos

The authors write (Appendix F): "a thorough discussion around
understanding how these models operate is outside the scope of the
current work, we hope these results encourage such investigations."
No existing work applies mechanistic interpretability to
iterative/recurrent transformers [7]. This is the contribution gap.

Every experiment in M2 must satisfy:
1. **Specificity**: name the model, checkpoint, layer range, data
   split, and evaluation mode.
2. **Variable isolation**: one independent variable per experiment.
3. **Falsifiability**: state what would disprove the hypothesis.
4. **Ontological purpose**: what truth do we learn, independent of
   engineering utility?
5. **Confounder control**: explicitly list and mitigate confounders.

### Reliability Discipline

The lecturer cautioned that interpretability tools may not give
"meaningful or reliable" results. Three practices address this:

1. **Triangulation rule**: every interpretive claim must be supported
   by at least two independent measurements. "Independent" means
   different protocols with different operational definitions, not
   the same metric computed twice. Claims supported by one measurement
   only are flagged as "single-measurement" and explicitly marked
   preliminary. See
   [§1.6](#16-reliability-discipline-and-triangulation) for the
   canonical M2 application with the cross-validation matrix.
2. **Non-claim list**: every protocol's output includes an explicit
   list of conclusions it does NOT support. This pre-empts
   over-claiming and protects against reviewer pushback.
3. **Tuned Lens as active triangulation partner**: Tuned Lens [24]
   runs alongside unweighted Logit Lens from the start of M2, not
   as a conditional fallback. Quantitative agreement thresholds are
   specified in
   [RQ1](#rq1-prediction-convergence-across-repeats-sq1).

### Ontological Cleanliness

Every research question answers the lecturer's "what is the question"
challenge before any code is written. The answer must be expressible
as: "the truth we seek is X; our measurement is reliable to extent Y;
the finding would be falsified by Z." Research questions that cannot
satisfy this template are not ready for implementation.

### Scope Discipline

M2 operates under a **2-week elapsed ceiling** from Abe's approval to
the M2 deliverable. Whatever protocols produce triangulated results by
the ceiling are reported; the remainder go to "future work". This
addresses the lecturer's concern that the analysis "could become an
entire project in its own right."

### Checkpoint Matrix

| ID | Architecture | Config | Steps | PPL | Purpose |
|----|-------------|--------|-------|-----|---------|
| C1 | CoT+Reserved | cotformer_full_depth, 2+21*5+1 | 40k | TBD | Baseline (no ln_mid, no depth_emb) |
| C2 | LN-CoT | cotformer_full_depth_lnmid_depthemb | 40k | 24.13 | Isolate ln_mid + depth_emb effect |
| C3 | LN-CoT | cotformer_full_depth_lnmid_depthemb | 60k | 23.59 | Effect of additional training |
| C4 | ADM v2 | adaptive_cotformer_..._single_final | 40k | TBD | Isolate router effect |
| C5 | ADM v2 | adaptive_cotformer_..._single_final | 60k | 24.06 | Router maturation |
| C4' | ADM v1 | Same architecture, pre-fix (no B4/B6) | 40k | -- | RNG divergence control |
| C5' | ADM v1 | Same architecture, pre-fix (no B4/B6) | 60k | -- | RNG divergence control |

Comparison C2-vs-C1 isolates `ln_mid`. C4-vs-C2 isolates the router.
C5-vs-C4 isolates training duration. C5'-vs-C5 isolates the
init-order RNG offset from the orphan-MoDBlock fix (see
`docs/reprod-notes.md` §C1 for the mechanism).

---

## 1. Milestone 2: Mechanistic Analysis

### 1.1 Ontological Framework

The central question is: **"What does iterative computation in
CoTFormer actually compute?"**

This decomposes into five specific sub-questions, each targeting a
distinct aspect of the model's internal dynamics:

| ID | Sub-Question | What Truth We Seek |
|----|-------------|-------------------|
| SQ1 | Does the model refine predictions monotonically across repeats, or does it restructure representations non-monotonically? | Whether repeats implement iterative refinement (converging toward a fixed point) or multi-phase processing (qualitative restructuring). This determines whether the "chain-of-thought" analogy is accurate or misleading. |
| SQ2 | Do attention heads specialise by repeat, and does this specialisation vary with layer depth within the mid-block? | Whether the weight-tied transformer blocks develop emergent functional specialisation despite identical parameters. If yes, the cross-repeat KV cache carries structurally different information at different repeats. |
| SQ3 | Does the ADM router learn a meaningful proxy for token difficulty, or does it approximate random pruning? | Whether the budget-adaptive claim is scientifically supported. The init-order RNG divergence (v1 vs v2 routing patterns; see `docs/reprod-notes.md` §C1) makes this urgent -- if the router converges to different patterns under minor training perturbations, it may not learn a robust difficulty signal. |
| SQ4 | When only a fraction of tokens survive to repeat 5, do the surviving tokens lose contextual information because their sequence neighbours halted earlier? | Whether the cross-repeat KV cache (containing mixed-depth frozen representations from halted tokens) provides meaningful context, or whether surviving tokens become informationally isolated. No existing literature addresses this. |
| SQ5 | Does the absence of a proper causal mask in the ADM's cross-repeat attention degrade routing quality? | Whether the ADM's routing decisions are less informed than they could be. The current mask (`k_indices <= indices`) enforces temporal ordering but does not account for repeat-depth relationships. |

### 1.2 Research Questions

Each sub-question generates specific, falsifiable research questions
with full variable isolation.

#### RQ1: Prediction Convergence Across Repeats (SQ1)

**Statement**: During evaluation on OWT2 validation data, does the
LN-CoTFormer (C3, 60k) converge monotonically toward the correct
next-token prediction across repeats 1 to 5, AND do the unweighted
Logit Lens and Tuned Lens [24] agree on the convergence trajectory?

- **IV**: Repeat index r in {1, 2, 3, 4, 5} (plus the `.5` halfway
  sub-sites from the activation capture framework)
- **DV**: Top-1 accuracy of `lm_head(ln_f(h_r))` (unweighted) and
  `lm_head(T_l(h_r))` (tuned), KL divergence
  `D_KL(P_r || P_{r-1})`, entropy `H(P_r)`
- **Controls**: Same checkpoint (C3), same data (OWT2 val), same
  forward pass
- **Confounders**:
  - `ln_mid` normalisation distorts the unweighted projection
    differently at each repeat (norm varies). Mitigation: dual
    measurement with and without `ln_mid` projection.
  - Early-repeat projections have a systematic baseline disadvantage
    because `lm_head` was trained only against the final residual.
    Mitigation: the trajectory SHAPE is what we measure, not absolute
    values; the disadvantage is constant across repeats.
  - **Geometric alignment confounder** (unique to unweighted lens):
    intermediate residuals live in a geometrically different subspace
    than final-layer residuals. Tuned Lens eliminates this via
    learned affine translation.
- **H0**: Top-1 accuracy is flat across repeats AND Spearman rho
  between lenses >= 0.80 on the trajectory.
- **Falsifies H0**: Paired t-test p < 0.01 between repeats 1 and 5
  (accuracy monotonic increase) AND the two lenses agree.
- **What truth**: the RATE of convergence reveals whether most useful
  work happens in repeats 1-2 (diminishing returns) or is distributed
  (each repeat adds substantial value).
- **Cross-checkpoint**: repeat on C1 (no `ln_mid`) and C5 (ADM) as
  separate invocations. C1-vs-C3 reveals whether `ln_mid` changes
  the convergence trajectory. C5-vs-C3 reveals whether the router
  alters which tokens converge early.

##### Tuned Lens triangulation

Agreement thresholds (from the operational specification):

- **Agree** if: Spearman rho between per-repeat top-1 accuracy
  sequences >= 0.80 AND sign agreement on all consecutive-repeat KL
  pairs AND cross-lens KL <= 0.5 nats averaged over sites.
- **Disagree** if: Spearman rho < 0.60 OR sign disagreement on >= 2
  consecutive pairs OR cross-lens KL > 1.0 nats OR top-1
  disagreement rate > 30 %.
- **Ambiguous** (0.60 <= rho < 0.80): report both trajectories, draw
  no strong conclusion.

Interpretation:

- Agreement: convergence trajectory is a genuine informational signal.
- Disagreement: interpretability measurement is inconclusive; fall
  back to non-lens RQ1 metrics (cosine distance between repeat hidden
  states, per-token CE loss) plus
  [RQ2](#rq2-representation-structural-similarity-sq1) (CKA) plus
  [RQ6](#rq6-effective-dimensionality-sq1) (effective
  dimensionality).

**Critical nuance**: Tuned Lens trains its affine translator to match
the model's own final-layer distribution, NOT the ground-truth next
token. "Convergence" under Tuned Lens means "convergence toward what
the final layer would have predicted." For the "does the model refine
toward the correct answer" interpretation, condition the trajectory
analysis on tokens where the final-layer probability of the
ground-truth token exceeds 0.5.

##### Tuned Lens failure modes (non-claims)

Three documented failure modes that Protocol A does NOT conclude
through:

1. Early-site predictions are inherently noisy -- report trajectory
   shape, do not over-interpret individual points.
2. No a priori "best layer" -- report the full trajectory across all
   21 mid-layers and 5 repeats.
3. "Overthinking" is a known phenomenon (intermediate layers
   outperform final) [24]. If convergence is non-monotonic, this is a
   model property, not a lens failure.

##### Tuned Lens training specification

Key parameters for `analysis/tuned_lens.py`:

| Parameter | Value |
|-----------|-------|
| Architecture | `nn.Linear(768, 768, bias=True)` (pure affine) |
| Init | `nn.init.eye_(weight)`, `nn.init.zeros_(bias)` |
| Loss | Forward KL: D_KL(p_final, p_lens) |
| Optimiser | SGD + Nesterov momentum, lr=1.0 (DEC-M2-017: fallback to Muon if convergence is poor) |
| Schedule | Linear decay over 250 steps |
| Translators | 105 (21 layers x 5 repeats) |
| Application site | After layer residual add, BEFORE `ln_mid` |
| Compute | ~5 h CPU for all translators |

Post-training diagnostic: inspect the singular-value spectrum of each
learned translator A_l. A sharp spectral drop cross-validates
[RQ6](#rq6-effective-dimensionality-sq1) dimensionality findings.

#### RQ2: Representation Structural Similarity (SQ1)

**Statement**: Across the 105 (layer, repeat) activation sites in the
LN-CoTFormer mid-block, which pairs produce structurally similar
representations as measured by centred kernel alignment (CKA)?

- **IV**: Pair of (layer_i, repeat_j) activation sites
- **DV**: CKA (linear + RBF kernels) over 2048 validation tokens;
  additionally, per-token cosine similarity
  `sim(x^(r), x^(r+1))` as a subsidiary metric measuring
  token-level representation decay across consecutive repeats
- **Controls**: Same checkpoint (C3), same data batch
- **Confounders**: Linear CKA saturates at high similarity for wide
  representations (mitigation: RBF kernel cross-check); finite-sample
  CKA has positive bias at small N (mitigation: Chun et al. [9]
  correction at 2048 tokens).
- **H0**: CKA > 0.9 across all cross-repeat pairs within each layer
  (redundant representations through same weights).
- **Falsifies H0**: CKA < 0.7 for any adjacent-repeat pair.
- **What truth**: if H0 holds, cross-repeat KV is redundant and
  highly compressible. If rejected, each repeat's contribution is
  unique. The layer-resolved CKA map shows WHERE the model does its
  most distinctive work.

#### RQ3: Attention Head Specialisation (SQ2)

**Statement**: In the LN-CoTFormer (C3), do attention heads partition
into functionally distinct classes based on cross-repeat attention
patterns, and does the partition differ between early (layers 3-7)
and late (layers 17-21) mid-layers?

- **IV**: (Layer index, head index) -- 252 heads
- **DV**: Final-token attention distribution at repeat 5 over all
  (token, repeat) pairs, clustered via k-means (k=4-6, silhouette
  selection); additionally, 7 attention-geometry metrics
  (`macro_budget`, `macro_rep_entropy`, `macro_in_entropy`,
  `macro_same_pos`, `head_budget`, `head_rep_entropy`,
  `appendix_f_heatmap`) per forward pass
- **Controls**: Same checkpoint, averaged over full validation set
- **Confounders**: Flash Attention does not return weights (mitigation:
  non-flash forward pass; verify overall PPL matches the flash-pass
  value to 4 decimal places).
- **H0**: Head-cluster assignments uniformly distributed across
  layers.
- **Falsification**: Layer-vs-cluster contingency chi-squared
  p < 0.01.
- **Ontological purpose**: determine whether the weight-tied
  transformer develops emergent functional specialisation despite
  identical parameters.
- **Implementation**: port the flash/causal switch and reshape
  pipeline as a runtime monkey-patch on
  `CausalSelfAttention.forward` -- no new model class, no
  checkpoint-loading mismatch. See
  [§1.7](#17-code-and-infrastructure-strategy) for the porting
  specification.

##### Zheng taxonomy gap

Zheng et al.'s taxonomy [7] assumes each head has a fixed layer
position and a fixed functional role. In our weight-tied model, the
SAME physical head appears at 5 "virtual layers" (repeats 1-5). A
head that functions as a Positional Head in repeat 1 may function as
an Induction Head in repeat 3 because input representations evolve
across repeats. This is genuinely novel territory -- no existing
taxonomy covers functional-role variation across repeats within the
same physical head.

##### Cross-repeat attention sinks

Xiao et al. [26] document attention sinks (disproportionate attention
mass on early tokens) in standard transformers. Our cross-repeat KV
cache creates novel sink opportunities: the first repeat's KV entries
are visible to all later repeats, potentially receiving
disproportionate attention from repeats 3-5. Protocol C explicitly
measures per-position attention mass aggregated across repeats to
detect such cross-repeat sinks.

#### RQ4: Router Validity (SQ3)

**Statement**: In the ADM (C5), do tokens halted at early repeats
have systematically lower next-token CE loss than tokens surviving to
repeat 5?

- **IV**: Token halting depth (repeat at which cascaded score drops
  below 0.5)
- **DV**: Per-token cross-entropy loss at final output; additionally,
  router-score vs state-delta correlation
  `corr(s_i, ||B(x^(i)) - x^(i)||)` measuring whether the router
  score tracks the magnitude of the block's actual transformation
- **Controls**: Same checkpoint (C5), same data (OWT2 val)
- **Confounders**: Position-in-sequence correlates with both loss and
  halting. Mitigation: partial correlation controlling for position.
- **H0**: No correlation (Pearson |r| < 0.1 after position control).
- **What truth**: if H0 rejected, router learns difficulty proxy. If
  H0 holds, router decisions are not meaningfully related to
  difficulty. C5'-vs-C5 tests robustness to training perturbations.

#### RQ5: Context Preservation Under Adaptive Depth (SQ4)

**Statement**: In the ADM (C5), do tokens surviving to repeat 5
produce higher-quality predictions when their context tokens also
survived deep, compared to when most context halted at repeat 1-2?

- **IV**: Fraction of context tokens surviving to same depth
- **DV**: Per-token CE loss for the target token
- **Controls**: Conditioned on target surviving to repeat 5
- **Confounders**: Sequence difficulty. Mitigation: z-score loss per
  sequence.
- **H0**: Target loss independent of context survival fraction.
- **What truth**: if H0 rejected, surviving tokens suffer "contextual
  isolation" -- a fundamental limitation of adaptive depth.

**Novelty**: no prior literature addresses contextual isolation in
adaptive-depth models. This investigation is novel with no existing
baselines or predictions to calibrate against -- both more valuable
as a contribution and less constrained by prior expectations.

#### RQ6: Effective Dimensionality (SQ1)

**Statement**: Does intrinsic dimensionality decrease across repeats
1 to 5 in the LN-CoTFormer (C3)?

- **IV**: Repeat index r in {1, 2, 3, 4, 5}
- **DV**: Participation ratio with finite-sample correction (Chun et
  al. [9]) at each (layer, repeat) site; additionally,
  residual-stream scale `||x||_2` per repeat as a pre-check for
  exponential norm growth
- **Controls**: Same checkpoint, same 2048 tokens
- **H0**: d_eff constant (+/- 5 %) across repeats within each layer.
- **Falsification**: Mann-Kendall monotonic decrease p < 0.01.
- **What truth**: if d_eff decreases, later repeats operate in
  lower-dimensional subspaces -- directly determining MLA rank
  allocation (DEC-EXT-001).

##### Rogue dimensions and three-way cross-validation

Compute participation ratio on raw hidden states AND after removing
the top-2 principal components. If the two diverge significantly,
rogue dimensions (outlier directions carrying disproportionate
variance) are present and must be flagged. Standard transformers show
an inverted-U pattern within each forward pass (d_eff increases to
mid-depth, decreases toward output; evidence from Chun et al. [9]
Figure 3b). Expect this within each repeat; across repeats, expect
monotonic decrease if iterative refinement compresses
representations.

Three independent rank metrics provide cross-validation:

| Metric | Source | What It Measures |
|--------|--------|-----------------|
| Chun gamma_both (participation ratio) | [9] | Effective dimensionality with finite-sample correction |
| Kobayashi pseudo-rank (95 % SV threshold) | [11] | Number of singular values above 5 % of max |
| KV-CoRE NER (Shannon entropy of normalised SVs) | [8] | Information-theoretic compressibility |

Agreement across all three strengthens claims; disagreement reveals
which aspect of the spectrum matters for KV compression.

#### RQ7: Interpolation-Mechanism Validity (SQ3)

**Statement**: In the ADM v2 (C5, 60k), when a token's router score
`s_i` is small (< 0.1), does the interpolated update
`x^(i+1) := (1 - s_i) x^(i) + s_i B(x^(i))` keep
`||x^(i+1) - x^(i)||` close to zero as the paper claims, or does
the persistence of near-duplicate hidden states cause attention at
later repeats to distribute mass across the clones, raising per-head
attention entropy on polluted positions?

- **IV**: Router score bin
  `s_i in {[0, 0.1), [0.1, 0.3), [0.3, 0.5), [0.5, 0.7), [0.7, 0.9), [0.9, 1.0]}`
- **DV-1**: Relative update magnitude
  `||x^(i+1) - x^(i)||_2 / ||x^(i)||_2`
- **DV-2**: Mean attention entropy at repeat 5 over the KV-cache
  entries whose source tokens had `s_{i-1} < 0.1` (the "clone set")
- **Controls**: Same checkpoint, same data, same forward pass, same
  seed
- **Confounders**: Position correlates with loss and halting
  (mitigation: partial correlation); easy-token bias (mitigation:
  control for hidden-state norm at entry).
- **H0**: DV-1 independent of `s_i` bin (|r| < 0.1) AND clone-set
  attention entropy <= 0.05 higher than non-clone set.
- **Falsification**: Attention entropy delta >= 0.2 (permutation
  test, 1000 shuffles).
- **Ontological purpose**: whether the ADM's interpolation formula is
  the "safety net" the paper [3] claims or a source of cross-repeat
  KV cache pollution. If positive, the 0.64 PPL gap reported in
  Section 5 of [3] gains a mechanistic explanation independent of the
  gradient-starvation hypothesis.
- **Priority**: HIGH -- feeds M3 mcHC design decision (router
  replacement vs complement).
- **Relation to [RQ4](#rq4-router-validity-sq3)**: orthogonal. RQ4
  measures input-side prediction-quality; RQ7 measures output-side
  mechanical behaviour. Both can be true, both false, or split.

##### Router evolution calibration

The CoTFormer paper (Section 5, [3]) shows that at 40k steps, most
tokens receive LOW `s_i` for the final repeat (KV cache pollution
maximal). At 60k, the distribution shifts rightward (more tokens use
deep processing). This implies: (a) the interpolation formula IS
being actively used, not bypassed; (b) our C5 (60k) should show the
rightward shift; (c) the clone-set entropy (DV-2) should be elevated
specifically at low-`s_i` positions.

Additionally, if K/V activations are low-rank (per
[RQ6](#rq6-effective-dimensionality-sq1) and Kobayashi et al. [11]),
representations across repeats become more similar. This creates
competing effects: beneficial (interpolation robust to weight errors)
vs detrimental (router cannot distinguish repeats, exacerbating the
clone problem). Protocol G's per-repeat-index rank measurement is
essential to disentangle these.

#### RQ8: Depth-Embedding Freeze Ablation (SQ1)

**Statement**: In the LN-CoTFormer (C3, 60k), if the learned depth
embedding is frozen to a constant index at evaluation time, how does
the per-repeat logit-lens distribution deviate from its natural state?

- **IV**: `depth_idx` condition in {actual, frozen-to-0, frozen-to-4,
  random} -- 4 conditions
- **DV-1**: Per-repeat logit-lens top-1 accuracy against ground truth
- **DV-2**: Per-repeat KL divergence
  `D_KL(P_{r, condition} || P_{r, actual})`
- **DV-3**: Per-repeat overall PPL
- **Controls**: Same checkpoint, same val batch, same forward-pass
  code path
- **Confounders**: `ln_mid` may mask changes in the underlying
  residual (mitigation: dual measurement without `ln_mid`);
  counterfactual probe limitations (mitigation: state findings as
  "the model's LEARNED use of `depth_emb` at inference").
- **H0**: Freezing has < 0.01 PPL impact per repeat AND < 1 % drop
  in per-repeat top-1 accuracy.
- **Falsification**: Frozen-to-0 causes > 0.05 PPL degradation at
  repeats 3-5 OR random condition causes non-monotonic accuracy
  fluctuations.
- **Ontological purpose**: whether `depth_emb` encodes meaningful
  "repeat identity"; directly informs M3 DEC-EXT-002 (MLA RoPE
  branch dimensioning).
- **Priority**: MEDIUM.

#### RQ9: Recurrent Inductive Bias for Algorithmic Counting

**Statement**: Does a 1-layer-4-repeat CoTFormer with weight tying
and recurrent attention learn to count from a random starting
integer, generalising to OOD sequence lengths (train <= 50,
test <= 200), at higher accuracy than a standard 4-layer Transformer
with RoPE trained on the same data with the same compute?

- **IV-1**: Architecture in {4-layer standard Transformer
  (depth-matched), 1-layer-4-repeat CoT+Reserved CoTFormer (no
  `ln_mid`, no `depth_emb`), 1-layer-4-repeat LN-CoTFormer (with
  `ln_mid`, no `depth_emb`)}
- **IV-2**: Test sequence length in {50, 100, 150, 200}
- **IV-3**: Counting task variant in {Task 1 sequential shifted-start
  (PRIMARY), Task 2 mod-10 (SECONDARY), Task 3 selective counting
  (SECONDARY)}
- **IV-4**: Model width n_embd in {128, 256, 512, 1024} (width sweep
  isolates width as a confounder; see
  [width sweep design](#width-sweep-design-dec-m2-012) below and
  DEC-M2-012 in [§1.9](#19-m2-decision-log))
- **DV**: Per-position next-token top-1 accuracy, per-sequence
  exact-match accuracy, accuracy-by-position curve
- **Controls**: 25-item checklist below
- **Confounders**: Parameter count (mitigation: the width sweep
  naturally provides param-matched comparisons -- see table below);
  RoPE confound (mitigation: both models use identical PE; if Task 1
  result is positive, the RECURRENCE overcomes RoPE's limit, not the
  absence of RoPE).
- **H0**: CoTFormer OOD accuracy at length 200 on Task 1 <= baseline
  OOD accuracy at length 200 (either baseline).
- **Falsification**: CoTFormer accuracy >= baseline + 5 pp, paired
  McNemar test p < 0.01, across >= 3 seeds.
- **Ontological purpose**: whether depth-recurrent architecture
  provides a sufficient inductive bias for algorithmic counting that
  depth-non-recurrent architectures lack. If positive, novel evidence
  that CoTFormer's vertical recurrence partially substitutes for RNN
  horizontal recurrence.

##### Merrill TC^0 framing

We empirically test whether vertical recurrence (weight-tied repeats)
provides an inductive bias analogous to horizontal chain-of-thought
for sequential reasoning tasks. Merrill and Sabharwal [17] prove that
horizontal CoT expands the complexity class from TC^0 to NC^1 with
O(n) decoding steps, but their proof requires generating intermediate
tokens that serve as external memory. Vertical recurrence updates the
hidden state in place without token generation, so the formal result
does not directly apply. Both the 1L-4R CoTFormer and the 4L standard
Transformer are O(1)-depth architectures -- both TC^0. Our experiment
tests whether the practical benefit transfers despite the formal gap.

##### Width sweep design (DEC-M2-012)

Chang and Bisk [25] use `n_embd=1024, n_head=8, d_ff=4096` for all
counting experiments. They never test below 1024-dim. At 1024-dim, 1L
and 2L Transformers fail; 4L is required. A single-point experiment
at `n_embd=128` risks producing an uninformative null result where
both models fail. The width sweep isolates width as a confounder:

| n_embd | d_k | d_ff | 4L params (approx.) | 1L-4R unique params |
|--------|-----|------|--------------------|--------------------|
| 128 | 32 | 512 | ~0.8M | ~0.2M |
| 256 | 64 | 1024 | ~3.2M | ~0.8M |
| 512 | 128 | 2048 | ~12.6M | ~3.1M |
| 1024 | 256 | 4096 | ~50M | ~12.6M |

**Natural param-matching via the sweep**: the sweep subsumes both
depth-matched and param-matched comparisons without additional runs:

| CoTFormer (unique params) | Approx. param-matches | Standard 4L at width |
|---------------------------|----------------------|---------------------|
| 1L-4R @ 256 (~0.8M) | ~ | 4L @ 128 (~0.8M) |
| 1L-4R @ 512 (~3.1M) | ~ | 4L @ 256 (~3.2M) |
| 1L-4R @ 1024 (~12.6M) | ~ | 4L @ 512 (~12.6M) |

**Reproduction check**: at n_embd=1024, our 4L Transformer is
architecturally identical to Chang and Bisk's [25]. If our 4L results
match theirs, our counting setup is validated. If they diverge, the
implementation requires debugging before any CoTFormer conclusions are
drawn.

**Mandatory validation gate** (DEC-M2-012a): for each width, confirm
the 4L baseline achieves >= 80 % IND accuracy on vanilla counting
before running the full CoTFormer comparison. If the baseline fails at
a given width, the comparison at that width is uninformative and is
reported as such.

**External citation flag**: the baseline design is inspired by Chang
and Bisk [25] (local copy: `docs/InductiveBiasCheng&Bisk.pdf`). Our
hyperparameters follow their published configuration where verifiable;
deviations are documented in [§1.9](#19-m2-decision-log).

##### RoPE fragility calibration

Expected 4L OOD accuracy based on Chang and Bisk [25]:

| Our test length | Ratio vs train | Expected 4L RoPE OOD accuracy |
|-----------------|----------------|-------------------------------|
| 50 (IND) | 1x | ~100 % |
| 100 | 2x | ~85-99 % |
| 150 | 3x | ~40-60 % (interpolated) |
| 200 | 4x | ~22-31 % |

##### 25-control checklist

| # | Control | Treatment |
|---|---------|-----------|
| 1 | Tokenisation | One-to-one integer-to-token; never compositional (DEC-M2-007) |
| 2 | Vocabulary size | Task-specific, matching Chang and Bisk [25] codebase: `vocab = [str(i) for i in range(max_output + 1)] + ['<pad>', 'a']`. For `tr50_te200`: V=203 (integers 0-200, pad, a). Token IDs follow enumeration order: 0="0", ..., 200="200", 201="<pad>", 202="a". No BPE/SentencePiece; the tokeniser is a trivial `w2i` dictionary constructed at init. Larger V wastes embedding capacity. |
| 3 | Placeholder `a` loss masking | Exclude positions where input=`a` from CE loss; compute CE only at output positions (DEC-M2-014b) |
| 4 | Train sequence length distribution | start ~ Uniform[0, V - L - 1], L ~ Uniform[1, 50]; fix generator seed |
| 5 | Per-integer frequency balance | Verify histogram of integer appearances as targets is uniform across [0, 199]; log before training |
| 6 | OOD test sequence lengths | {50, 75, 100, 125, 150, 175, 200} -- all held-out, never in training |
| 7 | Compute-matching convention | Both depth-matched and param-matched baselines via the width sweep (DEC-M2-010, DEC-M2-012) |
| 8 | Baseline hyperparameters | n_layer=4, n_head=4, n_embd in {128, 256, 512, 1024}, d_ff = 4 x n_embd |
| 9 | CoTFormer hyperparameters | n_repeat=4 (DEC-M2-015), min_repeat=4 (disable adaptive routing), depth_embedding=None, n_layer_begin=0, n_layer_end=0. **Two variants**: (a) CoT+Reserved (no `ln_mid`) and (b) LN-CoTFormer (with `ln_mid`). Variant (b) provides fair normalisation parity with the 4L baseline (which has pre-LN between layers). Both variants trained at all 4 widths. ADM excluded (adaptive routing is disabled for the counting task). |
| 10 | Positional encoding | RoPE for both (primary); NoPE condition deferred (DEC-M2-014a) |
| 11 | Optimiser + schedule | AdamW, lr=1e-4, weight_decay=0.1, beta1=0.9, beta2=0.95, cosine schedule with warmup\_steps=3000. Activation: **ReLU** (not GELU; matching Chang and Bisk codebase `config.py:27`). Gradient clipping: **max\_norm=0.3** (codebase `config.py:28`; NOT 1.0). `scale_attn_by_inverse_layer_idx=True` (GPT-2-specific attention regularisation). Fixed lr=1e-4 across all widths (matching [25]); if validation gate fails at n\_embd=128, re-run with lr=3e-4 as a secondary check. |
| 12 | Training step budget | 312.5K gradient steps at BS=32 (DEC-M2-016), matching Chang and Bisk [25]; report wall-clock separately |
| 13 | Batch size | batch_size=32, acc_steps=4 (effective 128) for both |
| 14 | Context length | sequence_length >= 200 for both at train and test time |
| 15 | Data leakage | Train/val/test from disjoint generator seeds; test includes (start, length) pairs not seen in train |
| 16 | Validation set | Fixed seed; use for checkpoint selection only |
| 17 | Loss function | Token-level CE, mean reduction over non-masked positions |
| 18 | Evaluation mode | Greedy argmax decoding; parallel next-token prediction (single forward pass, no autoregressive generation) |
| 19 | Reported metrics | Per-position top-1, per-OOD-length exact-match, per-task-variant accuracy breakdown |
| 20 | Seeds and reporting | >= 3 per config; **primary**: mean +/- SE across seeds; **secondary**: best-of-3 (for direct Chang and Bisk [25] comparison, who report best-of-5); McNemar's paired test p < 0.01 per OOD length (methodologically stronger than [25], who use no significance test) |
| 21 | Statistical test | McNemar's paired, p < 0.01 per OOD length |
| 22 | Pre-registration | H0 committed before any training run |
| 23 | Start-value distribution adequacy | Verify empirically that all integers in [0, 199] appear as targets at least N times |
| 24 | BOS token handling | Identical across both models; same token ID, same attention inclusion (DEC-M2-012b) |
| 25 | Fixed vs adaptive capacity | Fixed only (adaptive routing disabled via min_repeat=n_repeat=4) |

##### Additional controls from Chang and Bisk

- **BOS token** (DEC-M2-012b): include in sequence format. RoPE
  depends on BOS for certain counting tasks.
- **Shifted PEs** (DEC-M2-012c): randomise starting position 1..P
  for OOD-position robustness.
- **Loss at every position**: compute loss across all output
  positions, not just the last token.
- **Evaluation**: per-position top-1 accuracy, reported separately
  for IND (count <= 50) and OOD (count > 50). No sequence-level
  exact match.

##### Codebase implementation details (from `inductive_counting_with_LMs/`)

The Chang and Bisk reference codebase [25] (local copy:
`../inductive_counting_with_LMs/`) reveals implementation details the
paper does not state. These must be matched for the reproduction check
at n_embd=1024 to be valid.

**Data format** (from `data/rasp_primitives/counting_samesymbol_shiftedstart3__tr50_te200__/`):

```
Input:  ["k", "a", "a", ..., "a"]   (start integer k as FIRST token, then L 'a' tokens)
Output: ["-1", "k+1", "k+2", ...]   ("-1" = mask position, then count from k+1)
```

The starting integer k is an **explicit input token**, not inferred
from positional encoding. For the OOD test, start + L = max\_count
(200 for the te200 variant), so both start and length vary together.

**Tokeniser** (from `scripts/causal_transformer/dataset.py:45`):
a trivial string-to-index dictionary, NOT a BPE/SentencePiece
tokeniser. Constructed as `w2i = {w: i for i, w in enumerate(vocab)}`
where `vocab` is task-specific. Our `data/counting.py` replicates
this construction identically.

**Shifted PE mechanism** (from `dataset.py:10-12`):
`shift_value = random.randint(0, max_position_embeddings - seq_len)`,
then position IDs = `range(shift_value, shift_value + seq_len)`.
Contiguous integers from a random offset, not randomised.
`max_position_embeddings = 256` for the te200 task.

**Configuration divergences from standard GPT-2** (from `config.py`):

| Parameter | Chang and Bisk codebase | Standard GPT-2 | Our CoTFormer |
|-----------|------------------------|----------------|---------------|
| Activation | ReLU | GELU | GELU |
| Gradient clipping | max\_norm = 0.3 | 1.0 | 1.0 |
| `scale_attn_by_inverse_layer_idx` | True | False | False |
| Linear layer | Conv1D (transposed) | nn.Linear | nn.Linear |
| `tie_word_embeddings` | False | True (GPT-2) | True (hardcoded in `but_full_depth.py:216`; requires code change for the 4L baseline) |
| Dropout | 0.0 | 0.1 | 0.0 |

Our counting models adopt the Chang and Bisk configuration (ReLU,
grad\_clip=0.3, `scale_attn_by_inverse_layer_idx=True`) for the 4L
baseline to match their setup, and apply the same activation and
clipping to the CoTFormer variants for a fair comparison. The Conv1D
vs nn.Linear difference is mathematically equivalent (transposed
weight matrix) and does not affect model behaviour.

**Compute update**: with two CoTFormer variants (CoT+Reserved and
LN-CoTFormer) at all four widths, the run count increases from 24
to 36 (4 widths x 3 architectures x 3 seeds). Estimated wall-clock
on 2x L4: ~2.5-3 days (up from ~1.5-2 days).

#### RQ10: Table-2 Unconfound (DEFERRED)

Four-variant 60k training runs to isolate whether the paper's Table 2
PPL drops came from reduced reasoning space or from LayerNorm absence.
Cost: ~3-4 L4-days. Deferred post-MLA per DEC-M2-006. If M2 analysis
reveals that the `ln_mid` normalisation is the dominant contributor to
the C2-vs-C1 PPL gap (via the
[RQ1](#rq1-prediction-convergence-across-repeats-sq1) cross-checkpoint
comparison), this experiment gains priority.

### 1.3 Experimental Protocols

| Protocol | Serves | Implementation | Compute |
|----------|--------|---------------|---------|
| A: Logit Lens | [RQ1](#rq1-prediction-convergence-across-repeats-sq1) | `analysis/logit_lens.py` | ~10 min GPU |
| A-ext: Tuned Lens | [RQ1](#rq1-prediction-convergence-across-repeats-sq1) triangulation | `analysis/tuned_lens.py` | ~5 h CPU (training) + ~10 min CPU (application) |
| B: CKA Matrix | [RQ2](#rq2-representation-structural-similarity-sq1) | `analysis/cka.py` | ~20 min CPU |
| C: Attention Taxonomy | [RQ3](#rq3-attention-head-specialisation-sq2) | `analysis/attention_taxonomy.py` | ~9 min GPU |
| D: Router Analysis | [RQ4](#rq4-router-validity-sq3), [RQ5](#rq5-context-preservation-under-adaptive-depth-sq4) | `analysis/router_analysis.py` | ~10 min GPU |
| E: Residual Diagnostics | SQ1 | `analysis/residual_diagnostics.py` | ~5 min CPU |
| F: Effective Dimensionality | [RQ6](#rq6-effective-dimensionality-sq1) | `analysis/effective_dim.py` | ~5 min CPU |
| G: KV Compressibility | M3 design, [RQ6](#rq6-effective-dimensionality-sq1) cross-validation | `analysis/kv_rank.py` | ~25 min GPU |
| H: Interpolation Validity | [RQ7](#rq7-interpolation-mechanism-validity-sq3) | `analysis/interpolation_validity.py` | ~10 min GPU |
| I: Depth-Emb Freeze | [RQ8](#rq8-depth-embedding-freeze-ablation-sq1) | `analysis/depth_emb_freeze.py` | ~12 min GPU |

All protocols share a unified analysis framework under
`analysis/common/` with modular backends. Protocols A through I
share the same forward-pass activation cache to minimise GPU time.

The counting experiment
([RQ9](#rq9-recurrent-inductive-bias-for-algorithmic-counting)) is
not a protocol on existing checkpoints; it trains new models via
`iridis/counting-train/job.sh` and evaluates via
`iridis/counting-eval/job.sh`.

### 1.4 Theoretical Predictions

**Prediction 1** (weight-decay low-rank; from [11]): Weight decay
(wd=0.1) on weight-tied QK/OV matrices drives them to low-rank
solutions. The 5x repetition compounds this.

*Test*: Protocol G; three independent measurements:
- (a) Weight-matrix pseudo-rank of `W_K^T W_Q` and `PW_V`
  (Kobayashi [11] comparison)
- (b) Activation rank of K/V caches (KV compression relevance)
- (c) KV-CoRE NER of K/V caches [8] (information-theoretic
  complement)

*Two-tier threshold*: 95 % effective rank < 48 (fraction < 0.75 of
nominal d_head=64) is the literature-confirmed prediction, based on
Kobayashi et al.'s [11] direct measurements at wd=0.1 on a 125M
model with identical d_head=64. If rank < 40 is observed, this
constitutes evidence for an additional compounding effect from 5x
weight tying not studied in the original work.

**Prediction 2** (sparse attention from repetition; novel
hypothesis): We hypothesise that attention entropy decreases across
repeat indices, analogous to Zucchet et al.'s [12] finding that data
repetition accelerates sparse attention emergence. Unlike their
data-repetition setting, our weight-tied repetition provides
computational depth rather than statistical redundancy. This
prediction is novel and not directly calibrated against prior work.

*Test*: Protocol C; two complementary measurements:
1. Spearman rho < -0.5 (exploratory) across repeats 1-5
   (monotonicity test)
2. Paired comparison: repeat-1 entropy vs repeat-5 entropy (more
   powerful, tests whether the endpoint difference is significant
   regardless of trajectory shape)

### 1.5 Analysis Matrix

| Protocol | C1 | C2 | C3 | C4 | C5 | C4' | C5' | GPU? |
|----------|:--:|:--:|:--:|:--:|:--:|:---:|:---:|:----:|
| A: Logit Lens | X | X | X | X | X | -- | -- | Yes |
| A-ext: Tuned Lens | X | X | X | X | X | -- | -- | CPU |
| B: CKA | X | X | X | X | X | -- | -- | No |
| C: Attn Taxonomy | X | X | -- | X | -- | -- | -- | Yes |
| D: Router Analysis | -- | -- | -- | X | X | X | X | Yes |
| E: Residual | X | X | -- | X | -- | -- | -- | No |
| F: Eff. Dim | X | X | X | X | X | -- | -- | No |
| G: KV Compress. | X | -- | X | -- | X | -- | -- | Yes |
| H: Interp. Validity | -- | -- | -- | X | X | -- | -- | Yes |
| I: Depth-Emb Freeze | -- | -- | X | -- | -- | -- | -- | Yes |

Total GPU: ~2.3 h on L4. Total CPU (incl. Tuned Lens training):
~5.5 h.

### 1.6 Reliability Discipline and Triangulation

The lecturer raised two concerns at the M2 proposal stage. Each is
codified here as a constraint on the experimental programme:

**Concern 1 -- reliability**: interpretability tools may not give
"meaningful or reliable" results.

*Constraint*: every interpretive claim must be supported by at least
two independent measurements from the cross-validation matrix below.
"Independent" means different protocols with different operational
definitions, not the same metric computed twice. The
[RQ1](#rq1-prediction-convergence-across-repeats-sq1) Logit Lens /
Tuned Lens pair is the canonical example: the unweighted lens projects
through `lm_head(ln_f(h_r))` while the Tuned Lens applies a learned
affine translator trained on KL divergence from the final-layer
distribution. Agreement between the two establishes that the
convergence trajectory is a genuine informational signal rather than a
projection artefact.

| Claim | Primary | Secondary | Tertiary |
|-------|---------|-----------|----------|
| Representations refine across repeats | RQ1 Logit Lens | RQ1 Tuned Lens | RQ2 CKA decrease |
| KV cache is compressible | Protocol G activation rank | KV-CoRE NER per repeat | Chun PR ([RQ6](#rq6-effective-dimensionality-sq1)) |
| Weight tying induces low rank | Kobayashi weight-matrix rank [11] | Protocol G activation rank | Tuned Lens translator SV spectrum |
| Attention specialises by repeat | RQ3 k-means clustering | Zheng taxonomy mapping [7] | P2 entropy gradient |
| Router learns difficulty proxy | RQ4 halting-vs-loss | RQ7 update magnitude | Router-score vs state-delta |
| Recurrence helps counting | RQ9 accuracy sweep | OOD degradation rate | Cross-seed variance |

Convergence rule: "supported" if primary + >= 1 secondary agree;
"preliminary" if only primary; "contradicted" if primary and secondary
disagree.

**Concern 2 -- scope creep**: the analysis "could become an entire
project in its own right."

*Constraint*: the 2-week elapsed ceiling (see
[Scope Discipline](#scope-discipline)). Protocols that do not produce
triangulated results by the ceiling are moved to "future work"
regardless of completion status.

### 1.7 Code and Infrastructure Strategy

#### Layout

All M2 analysis code lives in a new `analysis/` top-level Python
package with informative script names (DEC-M2-002):

```
analysis/
├── __init__.py
├── common/
│   ├── __init__.py
│   ├── loader.py             load_model_from_checkpoint
│   ├── sites.py              ActivationSite enum + enumerate_sites
│   ├── collector.py          ActivationCollector (one forward per ckpt)
│   ├── data.py               iterate_owt2_val
│   ├── spectral.py           effective_rank, PR, kv_core_rank, NER
│   ├── stats.py              partial_correlation, paired_t_test
│   └── plotting.py           setup_figure, site_label, palette
├── logit_lens.py             RQ1 unweighted
├── tuned_lens.py             RQ1 tuned: translator training + triangulation
├── cka.py                    RQ2
├── attention_taxonomy.py     RQ3 (ports flash/causal switch + reshape)
├── router_analysis.py        RQ4 + RQ5
├── residual_diagnostics.py   SQ1 / Protocol E
├── effective_dim.py          RQ6
├── kv_rank.py                Protocol G / M3 design
├── interpolation_validity.py RQ7
├── depth_emb_freeze.py       RQ8
└── synthesis.py              Cross-checkpoint plotting
```

Multi-concern scripts (`router_analysis.py` hosts both
[RQ4](#rq4-router-validity-sq3) and
[RQ5](#rq5-context-preservation-under-adaptive-depth-sq4)) carry a
comment at the top pointing at the corresponding RQ block in this
document. The existing `analyze_kv_compression.py` is absorbed into
`analysis/kv_rank.py`.

A new `data/counting.py` module provides the synthetic counting
dataset generator for
[RQ9](#rq9-recurrent-inductive-bias-for-algorithmic-counting).

#### HPC packages

| Package | Purpose | Queue position |
|---------|---------|---------------|
| `iridis/analyze-lncot+adm/job-gpu.sh` | GPU protocols (~2.3 h) | 1st |
| `iridis/analyze-lncot+adm/job-cpu.sh` | CPU protocols + Tuned Lens training (~5.5 h) | 2nd |
| `iridis/counting-train/job.sh` | RQ9 baseline + CoTFormer training | 3rd |
| `iridis/counting-eval/job.sh` | RQ9 OOD evaluation across {50-200} | 4th |

#### Branch strategy (DEC-M2-009)

M2 develops entirely on `main`. The `origin/tak-ablation-code` branch
is ported as standalone Python modules, not cherry-picked commits.
Rationale: the tak branch is missing five critical DDP infrastructure
fixes (Gloo `cpu_group`, Gloo gather, atomic `/tmp`+`shutil.copy2`
save, `broadcast_buffers=False`, `NCCL_PROTO=Simple`) that main
carries. Switching would either accept DDP hang risk or require
forward-porting all five fixes.

**Specific ports**:

| Source (tak branch) | Target | What |
|---------------------|--------|------|
| `models/depthemb_eval_metrics.py:540-554` | `analysis/logit_lens.py` | JSD projection via `lm_head(ln_f(h_r))` |
| `models/depthemb_eval_metrics.py:374-512` | `analysis/common/collector.py` | Replaced by forward-hook collector |
| `models/fixed_cot_attn.py:152-172` | `analysis/attention_taxonomy.py` | Flash/causal switch (monkey-patch) |
| `models/fixed_cot_attn.py:383-426` | `analysis/attention_taxonomy.py` | Reshape pipeline + 7 attention metrics |

**Explicitly NOT ported** (with rationale):

- `distributed/ddp.py` 30-to-60 min timeout (tak flagged as TODO in
  commit `9a833b2`).
- `optim/base.py` `eval_steps = 24` hardcode (regression vs main's
  conditional).
- `optim/base.py` `get_raw_model` eval wrap (duplicate of main's B9
  chain).
- `models/__init__.py` `flash_train_caus_eval` import (broken -- file
  does not exist on tak's branch).
- `optim/modifed_base.py` (orphan with 0 importers, typo in
  filename).
- `models/tak_custom_cot.py` (superseded by `fixed_cot_attn.py`).

### 1.8 Compute Budget and Sequencing

#### M2 main analytic sub-stream

| Stage | Wall time |
|-------|-----------|
| Stage 1 flash collector (5 ckpts x 3 min) | ~15 min GPU |
| Stage 2 non-flash collector (3 ckpts x 5 min) | ~15 min GPU |
| Stage 3 ADM collector (2 ckpts x 5 min) | ~10 min GPU |
| logit_lens (5 ckpts x 2 min) | ~10 min GPU |
| cka (5 ckpts, CPU) | ~20 min CPU |
| attention_taxonomy (3 ckpts x 3 min) | ~9 min GPU |
| router_analysis RQ4+RQ5 (2 ckpts x 5 min) | ~10 min GPU |
| residual_diagnostics (5 ckpts, CPU) | ~5 min CPU |
| effective_dim (5 ckpts, CPU) | ~5 min CPU |
| kv_rank (5 ckpts x 5 min) | ~25 min GPU |
| interpolation_validity / RQ7 (2 ckpts x 5 min) | ~10 min GPU |
| depth_emb_freeze / RQ8 (1 ckpt x 4 conditions x 3 min) | ~12 min GPU |
| tuned_lens training (5 ckpts x 11 sites x ~5 min) | ~5 h CPU |
| tuned_lens application + triangulation | ~10 min CPU |
| synthesis | ~5 min CPU |
| **Subtotal GPU** | **~2.3 h** |
| **Subtotal CPU** | **~5.5 h** |

Split into two HPC submissions:
- `iridis/analyze-lncot+adm/job-gpu.sh` -- 1 GPU, 4 h walltime,
  runs Stages 1-3 plus GPU protocols.
- `iridis/analyze-lncot+adm/job-cpu.sh` -- 0 GPU, 8 h walltime,
  runs CPU protocols plus Tuned Lens translator training. Consumes
  the workspace activation cache from `job-gpu`.

#### M2 counting sub-stream

With the width sweep (DEC-M2-012) and both CoTFormer variants
(CoT+Reserved and LN-CoTFormer), the counting sub-stream trains
4 widths x 3 architectures x 3 seeds = 36 runs at 312.5K steps
each (DEC-M2-016):

| Component | Wall time |
|-----------|-----------|
| Dataset generation | ~10 min CPU |
| Training: 36 runs (small widths fast; 1024-dim dominates) | ~2.5-3 days on 2x L4 |
| OOD evaluation x 7 lengths x 36 configs | ~9 h L4 |
| Analysis + plotting | ~3 h CPU |
| **Total** | **~3-4 days on 2x L4** |

The 1024-dim runs dominate compute (~24-36 h each) but are essential
for the reproduction check against Chang and Bisk [25]. Small widths
(128, 256) complete in under 1 h each and can be parallelised on a
single L4.

#### Sequencing (linear, no parallelism)

1. Phase 0 cleanup (1 day): documentation and plan updates.
2. Phase 1 skeleton refactor (1 day): rename package, create
   `analysis/common/`, scaffold protocol scripts.
3. Phase 2 body implementation (2-3 days, 2 parallel orchs):
   P-A logit_lens + tuned_lens + cka + residual_diagnostics +
   effective_dim; P-B attention_taxonomy + router_analysis + kv_rank
   + interpolation_validity + depth_emb_freeze; P-C synthesis +
   integration tests.
4. Phase 3 counting sub-stream (1-2 days code):
   `iridis/counting-train/` + `iridis/counting-eval/` +
   `data/counting.py`.
5. Phase 4 execution (4-5 days HPC, linear queue):
   job-gpu -> job-cpu -> counting-train -> counting-eval.
6. Phase 5 synthesis and writeup (3-5 days).

**Total elapsed**: ~10-12 working days from approval, within the
2-week ceiling.

### 1.9 M2 Decision Log

| ID | Decision | Value |
|---|---|---|
| DEC-M2-000 | RQ roster | RQ1-9 IN, RQ10 deferred post-MLA, Tuned Lens IN as triangulation partner |
| DEC-M2-001 | RQ7/RQ8 scripts | Separate scripts (`interpolation_validity.py`, `depth_emb_freeze.py`) |
| DEC-M2-002 | Naming convention | Informative names, no `protocol_*` prefix; multi-concern scripts carry a comment pointing at this document |
| DEC-M2-003 | Code waves | Split (skeleton orch then body orchs) |
| DEC-M2-004 | RQ9 ownership | Abe executes, Aras reviews, sequential queue after M2 main |
| DEC-M2-005 | Attribution cleanup | Withdrawn (Aras = Tevfik Kavuncu = tak1e25, same person) |
| DEC-M2-006 | RQ10 | Defer post-MLA |
| DEC-M2-007 | Counting vocab | One-to-one V=256; compositional backlog with RQ10 consideration |
| DEC-M2-008 | Tuned Lens | Active from start, two-file split, quantitative thresholds, final-layer-correctness conditioning |
| DEC-M2-009 | Branch strategy | Strategy C -- develop on main, port tak's algorithms as Python modules |
| DEC-M2-010 | Counting compute-match | Both depth-matched and param-matched baselines; subsumed by DEC-M2-012 width sweep |
| DEC-M2-011 | Counting step budget | Match gradient steps; report wall-clock separately |
| DEC-M2-012 (REVISED) | Counting model width | n_embd sweep {128, 256, 512, 1024}, n_head=4, d_ff=4 x n_embd |
| DEC-M2-012a | Validation gate | 4L baseline >= 80 % IND accuracy before CoTFormer comparison per width |
| DEC-M2-012b | BOS token | Include in sequence format (RoPE depends on BOS) |
| DEC-M2-012c | Shifted PEs | Randomise starting position for OOD-position robustness |
| DEC-M2-013 | Counting primary task | Task 1 (shifted-start sequential) primary; Tasks 2-3 secondary |
| DEC-M2-014a | Counting NoPE condition | Defer |
| DEC-M2-014b | Counting start-token loss | Mask out (exclude trivial first-output from CE) |
| DEC-M2-015 | n_repeat for RQ9 | 4 (depth-matched to 4L); repeat=5 backlogged |
| DEC-M2-016 | Training steps | 312.5K at BS=32, matching Chang and Bisk [25] |
| DEC-M2-017 | Tuned Lens fallback | SGD first, Muon if convergence poor |
| DEC-M2-018 | RQ9 CoTFormer variants | Both CoT+Reserved (no `ln_mid`) and LN-CoTFormer (with `ln_mid`) at all widths. ADM excluded (routing disabled). 36 total runs |
| DEC-M2-019 | RQ9 baseline config | Match Chang and Bisk codebase: ReLU activation, grad\_clip=0.3, `scale_attn_by_inverse_layer_idx=True`, task-specific vocabulary (V=203 for te200) |

#### Backlog items (deferred post-M2-main)

- **repeat=5 counting**: reproduce
  [RQ9](#rq9-recurrent-inductive-bias-for-algorithmic-counting) with
  n_repeat=5 to align with the original 24-layer CoTFormer's 5-repeat
  architecture.
- **Compositional vocabulary**: test multi-digit tokenisation (each
  digit = one token) as a follow-up to the whole-number tokenisation
  baseline. Compositional tokenisation increases effective sequence
  length (e.g., "137" becomes three tokens ["1", "3", "7"]), which
  expands the attention window and may reveal attention specialisation
  absent in the short-sequence whole-number setting. This also
  enables counting to arbitrarily large integers (no fixed vocabulary
  ceiling), providing a stronger OOD generalisation test. Scope to
  be defined after the primary RQ9 results are analysed.

#### Novel contributions confirmed

| Contribution | Evidence |
|---|---|
| First mechanistic interpretability study of iterative/recurrent transformers | Belrose [24], Zheng [7], Chen [8]: no weight-tied models. Mohtashami [3] Appendix F: explicitly defers |
| First Tuned Lens on weight-tied architecture | Belrose [24]: all models have distinct params per layer |
| First KV-CoRE on cross-repeat cache | Chen [8]: all models are standard multi-layer |
| First counting experiment with CoTFormer | Chang and Bisk [25]: no weight-tied. CoTFormer [3]: no synthetic tasks |
| Merrill formal gap: vertical != horizontal CoT | Merrill [17]: proof requires token generation |

---

## 2. Milestone 3: Architectural Extensions

Each extension is motivated by specific M2 findings. Implementation details
are deliberately brief -- they will be expanded once M2 results inform design.

### 2.1 Multi-Head Latent Attention (MLA)

#### Motivation

CoTFormer's cross-repeat KV cache grows as `R*T` per mid-layer (4.4x overhead
at R=5). MLA [1] compresses KV through a low-rank latent bottleneck: 224
values/token instead of 1536 (6.9x compression). Savings scale with R.

Alternatives (GQA [2], KV quantization, sparse attention, fewer repeats) are
either less expressive, orthogonal, or change the compute budget rather than
efficiency. MLA preserves full per-head diversity and the complete attention
pattern while compressing the cache.

#### Design Decisions (provisional, pending M2)

| ID | Decision | Contingent On |
|----|----------|---------------|
| DEC-EXT-001 | Rank allocation: uniform kv_lora_rank=192 | RQ6. If d_eff varies >2x, adopt CARE [14]. |
| DEC-EXT-002 | Separate RoPE branch (d_R=32) | MHA2MLA [15] analysis |
| DEC-EXT-003 | RMSNorm on compressed latents | Standard [5] |
| DEC-EXT-004 | Mid-blocks only (prefix/suffix keep MHA) | No cache growth there |
| DEC-EXT-005 | Train from scratch, 40k steps, matched hyperparams | Fair comparison |
| DEC-EXT-006 | Cache compressed latents, decompress on access | Core MLA [1] |

Target: PPL < +0.5 vs LN-CoT, 6.9x mid-block KV compression, peak VRAM < 12 GB.

See Appendix A for projection architecture and dimensionality reference.

### 2.2 Manifold-Constrained HyperConnections (mcHC)

mcHC [6] generalises residual connections via multi-stream routing (`alpha`
streams, learned `C in R^{alpha x alpha}` per layer, Stiefel-constrained).
For CoTFormer: different subspaces routed through different repeats.

**Decision gate** (from M2): strong RQ4 correlation = mcHC as complement;
weak = mcHC as router replacement. CKA collapse (RQ2) = strong motivation.

Design: alpha=2, mid-layers only. Memory: 2x hidden state. Params: 8/layer.

### 2.3 Causal Cross-Repeat Mask for ADM

The ADM uses `is_causal=False` with an index-based mask (`k_indices <= indices`)
that enforces temporal ordering but ignores repeat-depth structure. A token at
repeat 3 attends uniformly to ALL previous tokens at ALL repeats.

**Research question** (SQ5): does adding repeat-depth-aware causal constraints
improve the perplexity-compute trade-off? Ablation: same ADM architecture,
modified mask, identical hyperparameters.

Design pending M2 Protocol C (attention taxonomy) and literature findings.

### 2.4 Residual Attention

Instead of compressing the full accumulated KV cache (grows linearly with R),
attend directly to per-position residual deltas across repeats. The "residual
cache" has size `R * d_model` per position (depth dimension), not `R * T *
d_model` (sequence times depth), avoiding linear growth entirely.

**Research question**: does attending to residual deltas preserve cross-repeat
information quality while eliminating the KV growth problem?

Design pending targeted literature review on differential/residual attention.

---

## Appendix A: MLA Dimensionality Reference

### Projection Architecture

```python
class MLACausalSelfAttention(nn.Module):
    def __init__(self, config):
        d = config.n_embd          # 768
        n_h = config.n_head        # 12
        d_c = config.kv_lora_rank  # 192
        d_qc = config.q_lora_rank  # 384
        d_R = config.qk_rope_head_dim  # 32
        d_nope = config.qk_nope_head_dim  # 64
        d_v = config.v_head_dim    # 64

        # Query: down -> norm -> up + RoPE branch
        self.wq_a = nn.Linear(d, d_qc, bias=False)
        self.q_norm = RMSNorm(d_qc)
        self.wq_b = nn.Linear(d_qc, n_h * d_nope, bias=False)
        self.wq_rope = nn.Linear(d_qc, n_h * d_R, bias=False)

        # KV: joint down -> norm -> up
        self.wkv_a = nn.Linear(d, d_c + d_R, bias=False)
        self.kv_norm = RMSNorm(d_c)
        self.wkv_b = nn.Linear(d_c, n_h * (d_nope + d_v), bias=False)

        # Output
        self.c_proj = nn.Linear(n_h * d_v, d, bias=False)
```

### RMSNorm

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight
```

### Dimensionality Map

```
Standard MHA:  Cache/token = 2 * 12 * 64 = 1536 floats
MLA:           Cache/token = 192 + 32 = 224 floats
Compression:   6.9x
```

---

## References

[1] DeepSeek-AI. *DeepSeek-V2: A Strong, Economical, and Efficient
    Mixture-of-Experts Language Model.* arXiv:2405.04434, 2024.

[2] J. Ainslie et al. *GQA: Training Generalized Multi-Query Transformer
    Models from Multi-Head Checkpoints.* arXiv:2305.13245, 2023.

[3] A. Mohtashami et al. *CoTFormer: A Chain-of-Thought Driven Architecture
    with Budget-Adaptive Computation Cost at Inference.* In ICLR, 2025.

[4] N. Shazeer. *Fast Transformer Decoding: One Write-Head is All You Need.*
    arXiv:1911.02150, 2019.

[5] B. Zhang and R. Sennrich. *Root Mean Square Layer Normalization.* In
    NeurIPS, 2019.

[6] DeepSeek-AI. *DeepSeek-V3 Technical Report.* arXiv:2412.19437, 2024.

[7] Z. Zheng et al. *Attention Heads of Large Language Models: A Survey.* In
    Patterns (Cell Press), 2025. arXiv:2409.03752.

[8] J. Chen et al. *KV-CoRE: Benchmarking Data-Dependent Low-Rank
    Compressibility of KV-Caches in LLMs.* arXiv:2602.05929, 2026.

[9] C. Chun et al. *Estimating Dimensionality of Neural Representations from
    Finite Samples.* arXiv:2509.26560, 2025.

[10] T. Nait Saada et al. *Mind the Gap: A Spectral Analysis of Rank Collapse
     and Signal Propagation in Attention Layers.* arXiv:2410.07799, 2025.

[11] S. Kobayashi et al. *Weight Decay Induces Low-Rank Attention Layers.*
     arXiv:2410.23819, 2024.

[12] N. Zucchet et al. *The Emergence of Sparse Attention: Impact of Data
     Distribution and Benefits of Repetition.* arXiv:2505.17863, 2025.

[13] Y. Xu et al. *Tracking the Feature Dynamics in LLM Training: A
     Mechanistic Study.* arXiv:2412.17626, 2024.

[14] Z. Zhou et al. *CARE: Covariance-Aware and Rank-Enhanced Decomposition
     for Enabling Multi-Head Latent Attention.* arXiv:2603.17946, 2026.

[15] T. Ji et al. *MHA2MLA: Towards Economical Inference -- Enabling
     DeepSeek's Multi-Head Latent Attention in Any Transformer.* In ACL, 2025.
     arXiv:2502.14837.

[16] J. Geiping et al. *Scaling up Test-Time Compute with Latent Reasoning: A
     Recurrent Depth Approach.* arXiv:2502.05171, 2025.

[17] W. Merrill and A. Sabharwal. *The Expressive Power of Transformers with
     Chain of Thought.* In ICLR, 2024. arXiv:2310.07923.

[18] S. Hao et al. *Training Large Language Models to Reason in a Continuous
     Latent Space (Coconut).* arXiv:2412.06769, 2024.

[19] B. Wang et al. *Grokked Transformers are Implicit Reasoners.* In
     NeurIPS, 2024. arXiv:2405.15071.

[20] H. Yao et al. *Thin Keys, Full Values: Reducing KV Cache via
     Low-Dimensional Attention Selection.* arXiv:2603.04427, 2026.

[21] J. Sok et al. *Garbage Attention in LLMs: BOS Sink Heads and Sink-aware
     Pruning.* arXiv:2601.06787, 2026.

[22] D. Lesens et al. *KQ-SVD: Compressing the KV Cache with Provable
     Guarantees on Attention Fidelity.* arXiv:2512.05916, 2025.

[23] X. Chen et al. *Reasoning Beyond Language: A Comprehensive Survey on
     Latent Chain-of-Thought Reasoning.* arXiv:2505.16782, 2025.

[24] N. Belrose et al. *Eliciting Latent Predictions from Transformers with
     the Tuned Lens.* In NeurIPS, 2023. arXiv:2303.08112.

[25] J. Chang and Y. Bisk. 2025. OpenReview `s3IBHTTDYl`. Local copy:
     `docs/InductiveBiasCheng&Bisk.pdf`.

[26] G. Xiao et al. *Efficient Streaming Language Models with Attention
     Sinks.* In ICLR, 2024. arXiv:2309.17453.
