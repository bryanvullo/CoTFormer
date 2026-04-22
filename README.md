# CoTFormer -- `abeprobes` experimentation branch

This is the **experimentation branch** of the CoTFormer reproduction
project. It carries all mechanistic analysis code, counting-substream
training and evaluation, and architectural-extension scaffolding that is
not yet ready for the stable reproduction branch.

For the straight reproduction of the original CoTFormer paper's Table 2
results (including Figure 4 and Section 5), switch to the `main` branch:

```bash
git checkout main
```

## Branch roles

| Branch | Purpose | Contents |
|---|---|---|
| `main` | Stable reproduction of the original CoTFormer paper | Training and evaluation packages for the paper's 24-layer models (Table 2 rows + Section 5 ADM). Supports the published perplexity and Figure 4 reproduction. No experimental variants, no mechanistic-analysis code. |
| `abeprobes` (this branch) | Experimentation and extension | Mechanistic-analysis `analysis/` package, counting substream (`data/counting.py` + `iridis/counting-*/`), architectural-extension proposals (MLA, mcHC), pre-registered research questions. |

The two branches are deliberately separated so the reproduction code on
`main` stays readable to external reviewers and collaborators without
being tangled with in-progress research. Changes flow one-way from
`main` to `abeprobes` via cherry-pick when a reproduction bug-fix needs
to be available in the experiment environment; they never flow from
`abeprobes` to `main` without an explicit promotion decision.

## Documentation

The documentation is split into three concerns:

| File | Concern | Primary audience |
|---|---|---|
| [`docs/reprod-notes.md`](docs/reprod-notes.md) | Divergences from the original paper, reproduction deltas, numerical comparison logs | Reproduction reviewers, co-authors |
| [`docs/extend-notes.md`](docs/extend-notes.md) | Scientific framing: research questions, hypotheses, experimental protocols, theoretical predictions, decision log | Research reviewers, co-authors |
| [`docs/extend-technical.md`](docs/extend-technical.md) | Implementation-level notes: package layout, BLOCKER resolutions, environment freeze, HPC deployment, branch discipline | Developers, agent orchs |

`extend-notes.md` and `extend-technical.md` are cross-referenced
bidirectionally; every scientific section in `extend-notes.md` has a
corresponding implementation section in `extend-technical.md`, and
every implementation section cites the scientific motivation.

## Current experimentation scope

Three coordinated work streams are in progress on this branch. Each is
self-contained in terms of file scope and runs in a dedicated `iridis/`
package.

### Mechanistic analysis (analytic sub-stream)

Code under `analysis/`. Ten experimental protocols (A through I plus a
calibration sub-protocol) across the CoTFormer checkpoint family
(`cotformer_full_depth`, `cotformer_full_depth_lnmid_depthemb`,
`adaptive_cotformer_..._single_final`). The protocols share a single
`ActivationCollector` that runs one forward pass per checkpoint and
caches per-site activations on `/scratch`, so CPU-only post-processors
(CKA, participation ratio, weight-level KV rank, synthesis) incur no
additional GPU cost beyond the shared pass.

Entry points live under `iridis/analyze-lncot+adm/` (not yet committed
at the time of writing; see `docs/extend-notes.md` §1.3 for the
protocol matrix and §1.8 for the compute budget).

### Counting substream

Code under `data/counting.py` (new synthetic dataset) and
`iridis/counting-{train,eval}/`. Tests whether weight-tied depth
recurrence provides an inductive bias for algorithmic counting that
a depth-matched standard transformer lacks. The experiment is
structured as a twin-pilot design:

- **Pilot 1**: strict Chang and Bisk [25] reproduction at `n_embd=1024`
  with the C&B codebase configuration (ReLU, grad_clip=0.3,
  scale_attn_by_inverse_layer_idx=True, tie_word_embeddings=False).
  Acts as a reproduction-fidelity control; if it fails to reach
  Chang and Bisk's published `te200` accuracy, the counting
  substream sheds.
- **Main sweep**: four CoTFormer variants (full_depth, reserved,
  lnmid, adm) at multiple widths using the CoTFormer-native training
  regime. Tests the architectural hypothesis under the training
  regime used elsewhere in this project.

See `docs/extend-notes.md` §1.2 RQ9 for the full experimental design
and §1.8 for the compute budget.

### Architectural extension scaffolding

Placeholder for Multi-Head Latent Attention (MLA), mcHC, causal
cross-repeat masking, and residual-attention extensions. These are
not implemented yet on `abeprobes`; they will be added in a
follow-up code wave after the mechanistic-analysis results inform
their design. See `docs/extend-notes.md` §2 for the provisional
design sketches.

## Environment and deployment

The shared Iridis X conda environment at
`/scratch/ab3u21/cotformer-env` is used on this branch just as on
`main`. Dependencies are pinned in
[`environment.yml`](environment.yml) and bit-exactly frozen in
`environment.lock.yml`. See
[`docs/extend-technical.md` §4](docs/extend-technical.md#4-environment-and-hpc-deployment)
for the HPC deployment protocol, conda rebuild procedure, and scratch
workspace layout.

The HPC cluster conventions inherited from `main` continue to apply:
self-submitting `job.sh` scripts, `iridis/env.sh` as the single source
of truth for environment variables, output-triage rule (large
intermediates on `/scratch`, only meaningful outputs in `run_N/`).
See the `main` branch README for the Iridis quickstart and SLURM
configuration references.

## Branch discipline

For contributors and agent orchs working on this branch:

- Verify `git branch --show-current` returns `abeprobes` before any
  edit. Never switch branches without an explicit instruction.
- Agents do not commit or push. All commits are authored manually.
- Cross-project paths are absolute; `git -C <abs-path>` is preferred
  over `cd`.
- Meta-tier agent infrastructure references (internal agent-memory
  paths, meta-memory filenames, internal mistake-ID codes) must not
  appear in project files. The firewall is bidirectional: the
  supervising agent-tier may read this branch, but this branch does
  not cite the agent-tier.

The full protocol lives in
[`docs/extend-technical.md` §5](docs/extend-technical.md#5-branch-discipline).

## Contributing

This branch is actively in flux. Pre-Phase 1 contributors should read
`docs/extend-notes.md` in full (the research questions and protocol
specifications are the contract for every code wave) and
`docs/extend-technical.md` (the engineering specifications).

## References

See the References section of
[`docs/extend-notes.md`](docs/extend-notes.md#references) for the full
bibliography (53 entries covering the CoTFormer paper, MLA/DeepSeek-V2,
Tuned Lens, CKA, participation ratio, KV-CoRE, attention sinks,
counting baselines, statistical methodology, and formal expressivity
results).
