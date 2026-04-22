"""Protocol A-ext -- Tuned Lens (Belrose et al. 2023).

Scope
-----
Addresses RQ1 triangulation. Trains per-(layer, repeat) affine
translators on the model's residual stream and reports
Tuned-Lens-projected top-1 accuracy, KL divergence, and entropy
alongside the unweighted logit-lens curves of Protocol A. The
translator-training recipe is the DEC-017 canonical Belrose 2023
setup: pure affine translator with identity initialisation, forward
KL loss against the model's own final-layer distribution, SGD with
Nesterov momentum at lr=1.0, linear decay over 250 steps, with a
Muon-optimiser fallback if the 95/105 convergence gate fails.

Falsifiability relevance
------------------------
The Tuned Lens is the reliability triangulation for RQ1: if the
unweighted logit-lens accuracy curve and the Tuned-Lens curve
disagree in monotonicity or in the paired-t-test verdict, the
convergence claim fails the `docs/extend-notes.md` §1.6 triangulation
bar (a single-measurement result is downgraded to preliminary).

Ontological purpose
-------------------
Removes the "lm_head only trained against the final residual"
systematic baseline disadvantage present in Protocol A: every
per-repeat projection is trained against its own residual, so the
early-repeat decoding is as fair as the late-repeat decoding. Any
systematic monotonicity that survives both the unweighted and the
tuned lens is a fact about the model's representations, not a fact
about lm_head bias.

Narrow novelty framing
----------------------
Per `docs/extend-notes.md` §1.2 RQ1 "Narrow novelty framing" and
DEC-023, the contribution is the first canonical Belrose 2023
[affine + bias, SGD + Nesterov, forward-KL, identity-init] Tuned
Lens on a CoTFormer-style weight-tied depth-recurrent transformer
with cross-repeat KV attention. The closest prior art is Paulo et
al. 2024, which transferred Tuned Lens to RNN architectures
(Mamba, RWKV) but not to weight-tied recurrent transformers.

Frozen specification (DEC-017, pre-registered in
docs/extend-notes.md §1.2 RQ1 "Tuned Lens training specification"):

- Architecture: ``nn.Linear(n_embd, n_embd, bias=True)`` per translator.
- Init: ``nn.init.eye_(layer.weight)`` and ``nn.init.zeros_(layer.bias)``.
- Loss: forward KL ``D_KL(p_final || p_lens)``.
- Optimiser: SGD + Nesterov momentum, lr=1.0.
- Schedule: linear decay over 250 steps.
- Translators: 21 layers x 5 repeats = 105 for C3.
- Application site: after layer residual add, BEFORE ``ln_mid``.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from analysis.common.loader import load_model_from_checkpoint
from analysis.common.plotting import savefig, setup_figure


def build_argparser() -> argparse.ArgumentParser:
    """Return the CLI parser for Protocol A-ext (Tuned Lens).

    Expected inputs: ``--checkpoint``, ``--checkpoint-file``,
    ``--workspace``, ``--output-dir``, ``--seed``,
    ``--translator-epochs``, ``--optimiser`` (``sgd-nesterov`` or
    ``muon``; DEC-017 fallback), ``--logit-lens-results``.
    """
    parser = argparse.ArgumentParser(
        description="Protocol A-ext -- canonical Belrose 2023 Tuned Lens"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Checkpoint directory containing summary.json + ckpt file",
    )
    parser.add_argument(
        "--checkpoint-file", type=str, default="ckpt.pt",
        help="Checkpoint filename within --checkpoint (default ckpt.pt)",
    )
    parser.add_argument(
        "--workspace", type=str, required=True,
        help="Workspace directory containing residuals produced by "
             "a prior analysis.logit_lens run",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for tuned_lens_translators.pt + "
             "tuned_lens_diagnostic.json + tuned_lens_triangulation.json",
    )
    parser.add_argument("--seed", type=int, default=2357)
    parser.add_argument(
        "--translator-epochs", type=int, default=250,
        help="SGD steps per translator (frozen at 250 per DEC-017)",
    )
    parser.add_argument(
        "--translator-batch-size", type=int, default=256,
        help="Tokens per SGD step (default 256, matching Belrose 2023 [24]). "
             "Reduce only when a compute-bound regime is documented via a "
             "dated DEC-NNN entry in docs/extend-notes.md §1.9.",
    )
    parser.add_argument("--translator-lr", type=float, default=1.0)
    parser.add_argument("--translator-momentum", type=float, default=0.9)
    parser.add_argument("--optimiser", type=str, default="sgd-nesterov",
                        choices=["sgd-nesterov", "muon"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--config-mode", type=str, default="raw",
                        choices=["raw", "argparse"])
    parser.add_argument("--module-path", type=str,
                        default="model.transformer.h_mid")
    parser.add_argument(
        "--logit-lens-results", type=str, default=None,
        help="Optional path to logit_lens_results.json; when provided, "
             "cross-lens KL and Spearman-rho triangulation is reported",
    )
    parser.add_argument(
        "--convergence-gate-sites", type=int, default=95,
        help="Minimum number of sites whose Tuned-Lens KL improves "
             "over unweighted lens at the same site (default 95 of 105)",
    )
    return parser


def _build_translator(n_embd: int, device: str, dtype: torch.dtype) -> nn.Linear:
    """Construct one canonical Belrose-2023 translator, identity-initialised.

    The kwargs ``eye_(weight)`` and ``zeros_(bias)`` are the exact
    initialisation from Belrose et al. 2023; at step 0 the translator
    is the identity map, so the lens prediction equals the unweighted
    logit-lens prediction at that site.
    """
    translator = nn.Linear(n_embd, n_embd, bias=True)
    nn.init.eye_(translator.weight)
    nn.init.zeros_(translator.bias)
    translator = translator.to(device=device, dtype=dtype)
    return translator


def _build_optimiser(
    translator: nn.Linear,
    optimiser_name: str,
    lr: float,
    momentum: float,
) -> torch.optim.Optimizer:
    """Construct the DEC-017 SGD+Nesterov optimiser or its Muon fallback.

    Muon is loaded from the ``muon_optimizer`` package (pinned in
    ``environment.yml`` per `docs/extend-notes.md` §1.7 HPC packages).
    If the package is unavailable we raise a ``RuntimeError`` rather
    than silently downgrading to SGD; DEC-017 mandates Muon as the
    specific fallback.
    """
    if optimiser_name == "sgd-nesterov":
        return torch.optim.SGD(
            translator.parameters(),
            lr=lr,
            momentum=momentum,
            nesterov=True,
        )
    if optimiser_name == "muon":
        try:
            from muon import Muon  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Muon fallback requested but `muon` package is not "
                "installed. Install via environment.yml per "
                "docs/extend-notes.md §1.7 HPC packages table."
            ) from exc
        return Muon(translator.parameters(), lr=lr, momentum=momentum)
    raise ValueError(f"_build_optimiser: unknown optimiser {optimiser_name!r}")


def _linear_decay_lambda(step: int, total_steps: int) -> float:
    """Linear decay from 1.0 at step 0 to 0.0 at ``total_steps``."""
    if total_steps <= 0:
        return 1.0
    ratio = max(0.0, 1.0 - step / float(total_steps))
    return float(ratio)


def _compute_p_final(
    h_final: torch.Tensor,
    ln_f: nn.Module,
    lm_head: nn.Linear,
    chunk_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (p_final, log_p_final) over ``h_final`` in ``chunk_size`` chunks.

    Avoids materialising the full ``(N_tokens, V)`` logits in one go;
    for V=50304 and N=2048 this is 400 MB (fits), but the chunking
    keeps memory pressure flat when we later scale to 8192 tokens
    for inter-batch CV.
    """
    n_tokens = h_final.shape[0]
    p_list = []
    log_p_list = []
    with torch.no_grad():
        for start in range(0, n_tokens, chunk_size):
            end = min(start + chunk_size, n_tokens)
            logits = lm_head(ln_f(h_final[start:end]))
            log_p = F.log_softmax(logits, dim=-1)
            p_list.append(log_p.exp())
            log_p_list.append(log_p)
    return torch.cat(p_list, dim=0), torch.cat(log_p_list, dim=0)


def _train_one_translator(
    h_site: torch.Tensor,
    p_final: torch.Tensor,
    log_p_final: torch.Tensor,
    ln_f: nn.Module,
    lm_head: nn.Linear,
    n_embd: int,
    device: str,
    dtype: torch.dtype,
    optimiser_name: str,
    lr: float,
    momentum: float,
    n_steps: int,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[nn.Linear, list[float]]:
    """Train one translator for ``n_steps`` with linear-decay SGD+Nesterov.

    Returns the trained translator and the per-step loss curve.
    """
    translator = _build_translator(n_embd, device=device, dtype=dtype)
    for param in translator.parameters():
        param.requires_grad_(True)

    optim = _build_optimiser(translator, optimiser_name, lr, momentum)
    loss_curve: list[float] = []

    n_tokens = h_site.shape[0]
    for step in range(n_steps):
        for group in optim.param_groups:
            group["lr"] = lr * _linear_decay_lambda(step, n_steps)

        idx = rng.choice(n_tokens, size=min(batch_size, n_tokens), replace=False)
        idx_torch = torch.from_numpy(idx).long().to(device=device)

        h_batch = h_site.index_select(0, idx_torch)
        p_final_batch = p_final.index_select(0, idx_torch)

        translated = translator(h_batch)
        lens_logits = lm_head(ln_f(translated))
        log_p_lens = F.log_softmax(lens_logits, dim=-1)

        # F.kl_div(log_input, target, reduction='batchmean') with
        # log_target=False computes mean over batch of sum_v
        # target * (log(target) - log_input) = KL(target || lens).
        loss = F.kl_div(log_p_lens, p_final_batch, reduction="batchmean", log_target=False)

        optim.zero_grad()
        loss.backward()
        optim.step()
        loss_curve.append(float(loss.item()))

    translator.eval()
    for param in translator.parameters():
        param.requires_grad_(False)
    return translator, loss_curve


def _identity_diagnostic(translator: nn.Linear) -> tuple[float, float, str]:
    """Compute ||A - I||_F, ||b||_2, and the §1.2 RQ1 diagnostic band.

    Returns ``(frobenius_weight, norm_bias, band)`` where ``band`` is
    one of ``"identity-equivalent"``, ``"mild-adjustment"``,
    ``"substantive-translation"``.
    """
    weight = translator.weight.detach().to(dtype=torch.float32).cpu()
    bias = translator.bias.detach().to(dtype=torch.float32).cpu()
    identity = torch.eye(weight.shape[0], dtype=weight.dtype)
    frob = float(torch.linalg.norm(weight - identity, ord="fro").item())
    bias_norm = float(torch.linalg.norm(bias).item())

    if frob < 0.1 and bias_norm < 0.1:
        band = "identity-equivalent"
    elif frob < 1.0:
        band = "mild-adjustment"
    else:
        band = "substantive-translation"
    return frob, bias_norm, band


def _sv_spectrum(translator: nn.Linear) -> list[float]:
    """Return the singular-value spectrum of the translator's weight.

    Used for the §1.2 RQ1 "Post-training diagnostic" cross-validation
    with RQ6 effective dimensionality.
    """
    weight = translator.weight.detach().to(dtype=torch.float32).cpu()
    sv = torch.linalg.svdvals(weight)
    return sv.numpy().tolist()


def _final_lens_kl_unweighted(
    h_site: torch.Tensor,
    p_final: torch.Tensor,
    log_p_final: torch.Tensor,
    ln_f: nn.Module,
    lm_head: nn.Linear,
    chunk_size: int = 256,
) -> float:
    """Unweighted-lens KL at ``h_site`` (identity translator as reference).

    The identity-translator path is ``lm_head(ln_f(h_site))``; this is
    the DEC-017 reference against which the trained Tuned Lens must
    improve for the convergence gate.
    """
    n_tokens = h_site.shape[0]
    total = 0.0
    with torch.no_grad():
        for start in range(0, n_tokens, chunk_size):
            end = min(start + chunk_size, n_tokens)
            lens_logits = lm_head(ln_f(h_site[start:end]))
            log_p_lens = F.log_softmax(lens_logits, dim=-1)
            kl = F.kl_div(log_p_lens, p_final[start:end], reduction="sum", log_target=False)
            total += float(kl.item())
    return total / max(1, n_tokens)


def _final_lens_kl_tuned(
    translator: nn.Linear,
    h_site: torch.Tensor,
    p_final: torch.Tensor,
    ln_f: nn.Module,
    lm_head: nn.Linear,
    chunk_size: int = 256,
) -> float:
    """Tuned-Lens KL at ``h_site`` after ``translator`` has been fit."""
    n_tokens = h_site.shape[0]
    total = 0.0
    with torch.no_grad():
        for start in range(0, n_tokens, chunk_size):
            end = min(start + chunk_size, n_tokens)
            translated = translator(h_site[start:end])
            lens_logits = lm_head(ln_f(translated))
            log_p_lens = F.log_softmax(lens_logits, dim=-1)
            kl = F.kl_div(log_p_lens, p_final[start:end], reduction="sum", log_target=False)
            total += float(kl.item())
    return total / max(1, n_tokens)


def _load_residuals_from_workspace(
    workspace: str,
    n_layer_mid: int,
    n_repeat: int,
    device: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, dict[tuple[int, int], torch.Tensor]]:
    """Load the final-layer residual and the 105 per-site residuals.

    Returns ``(h_final, residuals)`` where ``h_final`` is the
    ``(N_tokens, n_embd)`` PRE-``ln_f`` residual (the input to ``ln_f``)
    captured via ``ActivationSite.RESIDUAL_PRE_LN_F``, and
    ``residuals[(l, r)]`` is the ``(N_tokens, n_embd)`` pre-``ln_mid``
    residual at mid-block ``l`` of repeat ``r`` (1-indexed).

    The pre-``ln_f`` capture matches the canonical Belrose 2023 [24]
    Tuned Lens convention: the translator is trained to predict the
    pre-``ln_f`` residual, and ``_compute_p_final`` then applies
    ``ln_f`` exactly once before ``lm_head``. This removes the
    double-``ln_f`` approximation that the earlier POST-``ln_f``
    workaround required.
    """
    h_final_path = os.path.join(workspace, "residual_pre_ln_f.npy")
    if not os.path.exists(h_final_path):
        raise FileNotFoundError(
            f"load_residuals: workspace {workspace} missing residual_pre_ln_f.npy; "
            f"did analysis.logit_lens run with --workspace and request "
            f"the RESIDUAL_PRE_LN_F site? (Pre-hook captures the input to ln_f.)"
        )
    h_final_np = np.load(h_final_path)
    h_final = torch.from_numpy(h_final_np).to(device=device, dtype=dtype)

    residuals: dict[tuple[int, int], torch.Tensor] = {}
    for layer_idx in range(n_layer_mid):
        for repeat_idx in range(1, n_repeat + 1):
            path = os.path.join(
                workspace, f"residual_mid_l{layer_idx}_r{repeat_idx}.npy"
            )
            if not os.path.exists(path):
                continue
            arr = np.load(path)
            residuals[(layer_idx, repeat_idx)] = torch.from_numpy(arr).to(
                device=device, dtype=dtype
            )
    if not residuals:
        raise FileNotFoundError(
            f"load_residuals: workspace {workspace} has no "
            f"residual_mid_l{{L}}_r{{R}}.npy files; did analysis.logit_lens "
            f"run with the RESIDUAL_POST_MID site?"
        )
    return h_final, residuals


def _spearman_rho(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho between two rank sequences, NaN-tolerant."""
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a[mask]))
    rb = np.argsort(np.argsort(b[mask]))
    try:
        from scipy import stats as _scipy_stats

        rho, _ = _scipy_stats.spearmanr(a[mask], b[mask])
        return float(rho)
    except ImportError:
        ra_mean = float(np.mean(ra))
        rb_mean = float(np.mean(rb))
        num = float(np.sum((ra - ra_mean) * (rb - rb_mean)))
        denom = float(np.sqrt(np.sum((ra - ra_mean) ** 2) * np.sum((rb - rb_mean) ** 2)))
        return num / denom if denom > 0 else float("nan")


def _plot_identity_grid(
    frob_grid: np.ndarray,
    bias_grid: np.ndarray,
    output_dir: str,
) -> None:
    """Save the 21x5 identity-diagnostic heatmap as PNG."""
    fig, axes = setup_figure(1, 2, size=(14.0, 6.0))
    ax_frob, ax_bias = axes

    im_f = ax_frob.imshow(frob_grid, aspect="auto", origin="lower", cmap="viridis")
    ax_frob.set_xlabel("Repeat")
    ax_frob.set_ylabel("Mid-block layer")
    ax_frob.set_title("||A_l - I||_F per (layer, repeat)")
    fig.colorbar(im_f, ax=ax_frob)

    im_b = ax_bias.imshow(bias_grid, aspect="auto", origin="lower", cmap="viridis")
    ax_bias.set_xlabel("Repeat")
    ax_bias.set_ylabel("Mid-block layer")
    ax_bias.set_title("||b_l||_2 per (layer, repeat)")
    fig.colorbar(im_b, ax=ax_bias)

    os.makedirs(output_dir, exist_ok=True)
    savefig(fig, os.path.join(output_dir, "tuned_lens_identity_grid.png"))


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    model, config = load_model_from_checkpoint(
        checkpoint_dir=args.checkpoint,
        checkpoint_file=args.checkpoint_file,
        config_mode=args.config_mode,
        device=args.device,
        module_path=args.module_path,
    )
    n_embd = int(getattr(config, "n_embd"))
    n_repeat = int(getattr(model, "n_repeat", 1))

    mid_path = args.module_path
    if mid_path.startswith("model."):
        mid_path = mid_path[len("model.") :]
    cursor = model
    for name in mid_path.split("."):
        cursor = getattr(cursor, name)
    n_layer_mid = len(cursor)

    lm_head = model.lm_head.to(device=args.device).eval()
    ln_f = model.transformer.ln_f.to(device=args.device).eval()
    for param in lm_head.parameters():
        param.requires_grad_(False)
    for param in ln_f.parameters():
        param.requires_grad_(False)

    dtype = torch.float32  # translators train in FP32 on CPU per DEC-017
    h_final, residuals = _load_residuals_from_workspace(
        args.workspace, n_layer_mid, n_repeat, device=args.device, dtype=dtype
    )
    p_final, log_p_final = _compute_p_final(h_final, ln_f, lm_head)

    # First pass: SGD + Nesterov on all sites
    translators: dict[tuple[int, int], nn.Linear] = {}
    per_site_records: dict[tuple[int, int], dict[str, Any]] = {}

    for (layer_idx, repeat_idx), h_site in residuals.items():
        baseline_kl = _final_lens_kl_unweighted(
            h_site, p_final, log_p_final, ln_f, lm_head
        )

        translator, loss_curve = _train_one_translator(
            h_site=h_site,
            p_final=p_final,
            log_p_final=log_p_final,
            ln_f=ln_f,
            lm_head=lm_head,
            n_embd=n_embd,
            device=args.device,
            dtype=dtype,
            optimiser_name=args.optimiser,
            lr=args.translator_lr,
            momentum=args.translator_momentum,
            n_steps=args.translator_epochs,
            batch_size=args.translator_batch_size,
            rng=rng,
        )
        tuned_kl = _final_lens_kl_tuned(
            translator, h_site, p_final, ln_f, lm_head
        )
        frob, bias_norm, band = _identity_diagnostic(translator)
        sv = _sv_spectrum(translator)

        translators[(layer_idx, repeat_idx)] = translator
        per_site_records[(layer_idx, repeat_idx)] = {
            "baseline_unweighted_kl": baseline_kl,
            "tuned_kl": tuned_kl,
            "kl_improvement": baseline_kl - tuned_kl,
            "converged": bool(tuned_kl < baseline_kl),
            "frob_weight_minus_identity": frob,
            "bias_norm": bias_norm,
            "band": band,
            "sv_spectrum": sv,
            "loss_first": loss_curve[0] if loss_curve else float("nan"),
            "loss_last": loss_curve[-1] if loss_curve else float("nan"),
            "optimiser": args.optimiser,
        }

    # Convergence gate (DEC-017): >= 95 of 105 sites must improve
    n_sites = len(per_site_records)
    n_converged = sum(1 for record in per_site_records.values() if record["converged"])
    gate_threshold = args.convergence_gate_sites
    gate_pass = n_converged >= gate_threshold

    # Muon fallback: re-train failing sites when --optimiser was sgd-nesterov
    muon_retry_attempted = False
    if not gate_pass and args.optimiser == "sgd-nesterov":
        muon_retry_attempted = True
        for (layer_idx, repeat_idx), record in per_site_records.items():
            if record["converged"]:
                continue
            h_site = residuals[(layer_idx, repeat_idx)]
            try:
                translator, loss_curve = _train_one_translator(
                    h_site=h_site,
                    p_final=p_final,
                    log_p_final=log_p_final,
                    ln_f=ln_f,
                    lm_head=lm_head,
                    n_embd=n_embd,
                    device=args.device,
                    dtype=dtype,
                    optimiser_name="muon",
                    lr=args.translator_lr,
                    momentum=args.translator_momentum,
                    n_steps=args.translator_epochs,
                    batch_size=args.translator_batch_size,
                    rng=rng,
                )
                tuned_kl = _final_lens_kl_tuned(
                    translator, h_site, p_final, ln_f, lm_head
                )
                frob, bias_norm, band = _identity_diagnostic(translator)
                sv = _sv_spectrum(translator)

                if tuned_kl < record["baseline_unweighted_kl"]:
                    translators[(layer_idx, repeat_idx)] = translator
                    per_site_records[(layer_idx, repeat_idx)] = {
                        "baseline_unweighted_kl": record["baseline_unweighted_kl"],
                        "tuned_kl": tuned_kl,
                        "kl_improvement": record["baseline_unweighted_kl"] - tuned_kl,
                        "converged": True,
                        "frob_weight_minus_identity": frob,
                        "bias_norm": bias_norm,
                        "band": band,
                        "sv_spectrum": sv,
                        "loss_first": loss_curve[0] if loss_curve else float("nan"),
                        "loss_last": loss_curve[-1] if loss_curve else float("nan"),
                        "optimiser": "muon",
                    }
            except RuntimeError as exc:
                # Muon package missing: record the failure, do not retry.
                per_site_records[(layer_idx, repeat_idx)]["muon_fallback_error"] = str(exc)
                break
        n_converged = sum(1 for r in per_site_records.values() if r["converged"])
        gate_pass = n_converged >= gate_threshold

    if gate_pass:
        verdict = "PASS"
    elif n_converged >= int(0.75 * n_sites):
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    # Emit the 21 x 5 identity-diagnostic grid as arrays + PNG
    frob_grid = np.full((n_layer_mid, n_repeat), np.nan, dtype=np.float32)
    bias_grid = np.full((n_layer_mid, n_repeat), np.nan, dtype=np.float32)
    for (layer_idx, repeat_idx), record in per_site_records.items():
        frob_grid[layer_idx, repeat_idx - 1] = record["frob_weight_minus_identity"]
        bias_grid[layer_idx, repeat_idx - 1] = record["bias_norm"]

    # Triangulation with unweighted Logit Lens (optional input)
    triangulation: dict[str, Any] = {
        "logit_lens_results_provided": args.logit_lens_results is not None,
    }
    targets_path = os.path.join(args.workspace, "targets.npy")
    targets_np: np.ndarray | None = None
    if os.path.exists(targets_path):
        targets_np = np.load(targets_path)
    if args.logit_lens_results and os.path.exists(args.logit_lens_results):
        with open(args.logit_lens_results, "r") as fh:
            logit_lens = json.load(fh)
        try:
            unweighted_top1 = np.asarray(
                logit_lens["aggregate"]["mean_top1_lnf"], dtype=np.float64
            )
        except KeyError:
            unweighted_top1 = None

        tuned_top1 = np.full((n_layer_mid, n_repeat), np.nan, dtype=np.float64)
        cross_lens_kl = np.full((n_layer_mid, n_repeat), np.nan, dtype=np.float64)
        for (layer_idx, repeat_idx), record in per_site_records.items():
            translator = translators[(layer_idx, repeat_idx)]
            h_site = residuals[(layer_idx, repeat_idx)]
            with torch.no_grad():
                translated = translator(h_site)
                lens_logits_tuned = lm_head(ln_f(translated))
                lens_logits_unweighted = lm_head(ln_f(h_site))
                preds_tuned = lens_logits_tuned.argmax(dim=-1).cpu().numpy()
                log_p_tuned = F.log_softmax(lens_logits_tuned, dim=-1)
                log_p_unw = F.log_softmax(lens_logits_unweighted, dim=-1)
                p_tuned = log_p_tuned.exp()
                kl_cross = float(
                    (p_tuned * (log_p_tuned - log_p_unw)).sum(dim=-1).mean().item()
                )
            cross_lens_kl[layer_idx, repeat_idx - 1] = kl_cross
            if targets_np is not None:
                trim = min(len(targets_np), preds_tuned.shape[0])
                tuned_top1[layer_idx, repeat_idx - 1] = float(
                    (preds_tuned[:trim] == targets_np[:trim]).sum()
                ) / max(1, trim)

        triangulation["tuned_top1_shape"] = list(tuned_top1.shape)
        triangulation["tuned_top1"] = tuned_top1.tolist()
        triangulation["cross_lens_kl_grid"] = cross_lens_kl.tolist()
        triangulation["cross_lens_kl_mean"] = float(np.nanmean(cross_lens_kl))
        triangulation["cross_lens_kl_threshold_agree"] = 0.5
        triangulation["cross_lens_kl_threshold_disagree"] = 1.0
        if unweighted_top1 is not None and targets_np is not None:
            per_layer_rhos = []
            for layer_idx in range(min(n_layer_mid, unweighted_top1.shape[0])):
                rho = _spearman_rho(
                    unweighted_top1[layer_idx], tuned_top1[layer_idx]
                )
                per_layer_rhos.append(rho)
            mean_rho = float(np.nanmean(per_layer_rhos))
            triangulation["per_layer_spearman_rho"] = per_layer_rhos
            triangulation["mean_spearman_rho"] = mean_rho
            mean_cross_kl = float(np.nanmean(cross_lens_kl))
            if mean_rho >= 0.80 and mean_cross_kl <= 0.5:
                agreement = "agree"
            elif mean_rho < 0.60 or mean_cross_kl > 1.0:
                agreement = "disagree"
            else:
                agreement = "ambiguous"
            triangulation["agreement_band"] = agreement

    # Persist
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(
        {
            "translators": {
                f"l{layer_idx}_r{repeat_idx}": translator.state_dict()
                for (layer_idx, repeat_idx), translator in translators.items()
            },
            "n_layer_mid": int(n_layer_mid),
            "n_repeat": int(n_repeat),
            "n_embd": int(n_embd),
            "optimiser": args.optimiser,
        },
        os.path.join(args.output_dir, "tuned_lens_translators.pt"),
    )

    diagnostic = {
        "checkpoint": args.checkpoint,
        "checkpoint_file": args.checkpoint_file,
        "seed": args.seed,
        "optimiser": args.optimiser,
        "translator_epochs": args.translator_epochs,
        "translator_batch_size": args.translator_batch_size,
        "translator_lr": args.translator_lr,
        "translator_momentum": args.translator_momentum,
        "convergence_gate_threshold_sites": gate_threshold,
        "n_sites": int(n_sites),
        "n_converged": int(n_converged),
        "muon_retry_attempted": muon_retry_attempted,
        "verdict": verdict,
        "frob_identity_grid": frob_grid.tolist(),
        "bias_norm_grid": bias_grid.tolist(),
        "per_site": {
            f"l{layer_idx}_r{repeat_idx}": record
            for (layer_idx, repeat_idx), record in per_site_records.items()
        },
    }
    with open(os.path.join(args.output_dir, "tuned_lens_diagnostic.json"), "w") as fh:
        json.dump(diagnostic, fh, indent=2)

    with open(os.path.join(args.output_dir, "tuned_lens_triangulation.json"), "w") as fh:
        json.dump(triangulation, fh, indent=2)

    _plot_identity_grid(frob_grid, bias_grid, args.output_dir)


if __name__ == "__main__":
    main()
