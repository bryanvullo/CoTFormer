"""DV-4 -- Causal zero-ablation per repeat (RQ9).

Scope
-----
Addresses DV-4 of RQ9 per DEC-027. Loads a counting-task checkpoint,
runs the OOD test set twice per repeat ``r in {1, ..., n_repeat}``:

1. **Baseline forward**: no intervention; record OOD counting
   accuracy + cross-entropy.
2. **Ablated forward**: install a forward-hook on every
   ``h_mid[i]`` block that, when the model's per-forward repeat
   counter equals ``r``, replaces the block's OUTPUT with its INPUT
   (identity-replacement). The residual stream at repeat ``r`` passes
   through unchanged; the block's additive contribution (attn + MLP
   delta) is zeroed.

The per-repeat **causal-importance score** is
``baseline_accuracy - ablated_accuracy``; a high value indicates that
the model relies on repeat ``r`` for the counting prediction, a
low value (near zero) indicates the repeat is redundant for the
OOD test set.

Falsifiability relevance
------------------------
Per `docs/extend-notes.md` §1.2 RQ9 DV-4 row, the causal-importance
profile is the second structural DV triangulating the RQ9 main
effect. A "recurrent inductive bias for counting" claim (RQ9
positive) is strengthened by a monotone-by-repeat importance
profile with the final repeats dominating; a uniform profile would
weaken the interpretation.

Interpretation caveat
---------------------
Zero-ablation (here: identity-replacement) is NOT the same as
"removing the mechanism": it injects a specific signal (identity
preservation of the residual stream at that repeat). Standard
mechanistic-interpretability caveat documented in §1.2 RQ9 DV-4
"Implementation note"; the result answers "what happens if this
repeat does nothing" rather than "what does this repeat do".

Repeat-counter mechanism
------------------------
The script's hook structure mirrors
``analysis.common.collector.ActivationCollector`` counter
bookkeeping: a forward-pre-hook on the model resets the counter
at the start of every forward pass; a forward-post-hook on
``transformer.ln_mid`` (or the last ``h_mid`` block when
``ln_mid`` is absent -- e.g. the plain ``cotformer_full_depth``
class) increments the counter at the end of each repeat. Each
``h_mid`` block's forward-post-hook reads the counter at fire
time; when it matches the target ablation repeat, the hook returns
``inputs[0]`` (the pre-block residual) to make the block an
identity map.

Smoke test
----------
``python -m analysis.counting_dv4_causal --smoke-test`` verifies
the ablation arithmetic on a pure-numpy synthetic 4-repeat pipeline:
the "model" is a deterministic residual stream whose final token
equals the sum of per-repeat non-zero contributions; the test
confirms that (a) baseline accuracy on a fixed target is a known
value, (b) ablating any single repeat with a non-zero contribution
shifts the prediction, and (c) ablating a zero-contribution repeat
leaves accuracy unchanged.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np


OOD_LENGTH_RANGE = (51, 200)


def build_argparser() -> argparse.ArgumentParser:
    """Return the CLI parser for DV-4."""
    parser = argparse.ArgumentParser(
        description="DV-4 causal zero-ablation per repeat"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=False,
        help="Checkpoint directory containing summary.json + checkpoint file",
    )
    parser.add_argument(
        "--checkpoint-file", type=str, default="ckpt.pt",
        help="Checkpoint filename (default ckpt.pt)",
    )
    parser.add_argument(
        "--output-dir", type=str, required=False,
        help="Directory for counting_dv4_causal_results.json",
    )
    parser.add_argument("--seed", type=int, default=19937,
                        help="OOD test-set seed (project OOD convention)")
    parser.add_argument("--n-samples", type=int, default=500,
                        help="OOD sample count (default 500)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Per-batch sample count (default 8)")
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--config-mode", type=str, default="raw",
                        choices=["raw", "argparse"])
    parser.add_argument("--module-path", type=str,
                        default="model.transformer.h_mid")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run the synthetic-4-repeat smoke test and exit")
    return parser


# ---------------------------------------------------------------------------
# Ablation context manager (torch-dependent; inner function imports)
# ---------------------------------------------------------------------------


_COUNTER_ATTR = "_analysis_dv4_repeat_counter"


@contextmanager
def ablation_hooks(model: "Any", target_repeat: int, module_path: str) -> Iterator[None]:
    """Install identity-replacement hooks on every h_mid block for ``target_repeat``.

    Parameters
    ----------
    model : nn.Module
        Loaded CoTFormer / baseline model. Must expose
        ``transformer.h_mid`` and either ``transformer.ln_mid`` or a
        non-empty h_mid list (the counter increment-hook falls back
        to the last h_mid block when ln_mid is absent).
    target_repeat : int
        1-indexed repeat to ablate. When the model's per-forward
        counter equals ``target_repeat`` at a block's post-hook fire,
        the hook returns the block's input to make the block an
        identity map. ``target_repeat <= 0`` disables ablation
        (baseline pass); ``target_repeat > n_repeat`` raises.
    module_path : str
        Dotted path to the h_mid list. Default matches the CoTFormer
        layout; Arm A baseline also uses ``transformer.h_mid``.

    Yields
    ------
    None
        Control to the caller's forward loop. Hooks are installed
        on entry and removed on exit (via ``try/finally``).
    """
    # Deferred torch import: smoke-test mode does not need torch.
    import torch  # noqa: F401

    transformer = getattr(model, "transformer", None)
    if transformer is None:
        raise AttributeError(
            "ablation_hooks: model has no transformer attribute"
        )

    # Resolve the h_mid block list via the module_path suffix.
    mid_path = module_path
    if mid_path.startswith("model."):
        mid_path = mid_path[len("model.") :]
    cursor = model
    for name in mid_path.split("."):
        cursor = getattr(cursor, name)
    if not hasattr(cursor, "__len__") or len(cursor) == 0:
        raise RuntimeError(
            "ablation_hooks: model has an empty h_mid list at "
            f"{module_path!r}; cannot ablate."
        )
    h_mid = cursor

    n_repeat = int(getattr(model, "n_repeat", 1) or 1)
    if target_repeat > 0 and target_repeat > n_repeat:
        raise ValueError(
            f"ablation_hooks: target_repeat={target_repeat} exceeds "
            f"model.n_repeat={n_repeat}"
        )

    handles: list = []

    def _reset_counter(module, inputs):
        setattr(model, _COUNTER_ATTR, 1)

    def _increment_counter(module, inputs, output):
        current = int(getattr(model, _COUNTER_ATTR, 1))
        setattr(model, _COUNTER_ATTR, current + 1)

    def _make_ablation_hook():
        def _hook(module, inputs, output):
            # inputs is a positional tuple; inputs[0] is the residual
            # stream passed to the block. target_repeat <= 0 means
            # baseline pass; pass output through unchanged.
            if target_repeat <= 0:
                return output
            current = int(getattr(model, _COUNTER_ATTR, 1))
            # Clamp to [1, n_repeat] in case the increment hook has
            # already fired one too many times on the last block of
            # a repeat (mirrors ActivationCollector._make_site_hook).
            current = max(1, min(current, n_repeat))
            if current == target_repeat:
                # Identity-replacement: return the module's input
                # instead of the block's output so the residual
                # passes through unchanged.
                return inputs[0]
            return output

        return _hook

    # 1. Reset counter at the start of each forward pass.
    handles.append(model.register_forward_pre_hook(_reset_counter))

    # 2. Increment counter at the end of each repeat. Try ln_mid first;
    # fall back to the last h_mid block (same pattern as the collector).
    ln_mid = getattr(transformer, "ln_mid", None)
    if ln_mid is not None:
        handles.append(ln_mid.register_forward_hook(_increment_counter))
    else:
        last_mid = h_mid[len(h_mid) - 1]
        handles.append(last_mid.register_forward_hook(_increment_counter))

    # 3. Install the ablation hook on every h_mid block.
    for block in h_mid:
        handles.append(block.register_forward_hook(_make_ablation_hook()))

    setattr(model, _COUNTER_ATTR, 1)
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
        if hasattr(model, _COUNTER_ATTR):
            delattr(model, _COUNTER_ATTR)


# ---------------------------------------------------------------------------
# Accuracy computation shared between baseline and ablated runs
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class CountingRunMetrics:
    """Accuracy + cross-entropy over a single forward pass of the OOD set."""

    n_samples: int
    exact_match_accuracy: float
    per_position_accuracy: float
    cross_entropy: float


def _counting_metrics_from_outputs(
    logits: "Any",  # torch.Tensor
    targets: "Any",  # torch.Tensor
    loss_mask: "Any",  # torch.Tensor
) -> dict[str, float]:
    """Compute exact-match + per-position accuracy + CE from model outputs.

    Pure-torch helper used by ``run_one_pass`` on both baseline and
    ablated forwards. Exact-match accuracy = fraction of samples where
    ALL scored positions (loss_mask == 1) predict correctly;
    per-position accuracy = fraction of correct scored positions.
    """
    import torch  # noqa: F401
    import torch.nn.functional as F

    B, T, V = logits.shape
    preds = logits.argmax(dim=-1)
    correct = ((preds == targets) & (loss_mask > 0.5)).float()  # (B, T)
    total_pp = loss_mask.sum().clamp(min=1.0)
    per_position_accuracy = float((correct.sum() / total_pp).item())

    # Exact-match: a sample counts as correct iff every scored
    # position is correct. Compute the per-sample valid-count and the
    # per-sample correct-count; the sample is correct iff the two are
    # equal AND at least one position is scored.
    per_sample_valid = loss_mask.sum(dim=1)
    per_sample_correct = correct.sum(dim=1)
    sample_correct = (
        (per_sample_valid > 0.5) & (per_sample_correct == per_sample_valid)
    ).float()
    exact_match_accuracy = float(sample_correct.mean().item())

    per_pos_ce = F.cross_entropy(
        logits.reshape(-1, V), targets.reshape(-1),
        reduction="none", ignore_index=-1,
    ).reshape(B, T)
    masked_ce = per_pos_ce * loss_mask
    ce = float((masked_ce.sum() / total_pp).item())

    return {
        "exact_match_accuracy": exact_match_accuracy,
        "per_position_accuracy": per_position_accuracy,
        "cross_entropy": ce,
    }


def run_one_pass(
    model,
    loader,
    device: str,
    target_repeat: int,
    module_path: str,
    accepts_attention_mask: bool,
) -> CountingRunMetrics:
    """Run ONE forward pass over the OOD loader with ``target_repeat`` ablated.

    Parameters
    ----------
    target_repeat : int
        0 or negative disables ablation (baseline). Positive values
        trigger identity-replacement on every h_mid block at that
        repeat.
    accepts_attention_mask : bool
        ``True`` for the Arm A baseline (``but_full_depth``) which
        has a ``--attention_mask`` kwarg. ``False`` for all CoTFormer
        variants as of DIR-001 (the kwarg would raise TypeError).
    """
    import torch  # noqa: F401

    running = {
        "n_samples": 0, "per_position_correct": 0.0, "per_position_total": 0.0,
        "exact_match_correct": 0.0, "exact_match_total": 0, "ce_sum": 0.0,
        "ce_total": 0.0,
    }

    model.eval()
    ctx = ablation_hooks(model, target_repeat, module_path)
    with torch.no_grad(), ctx:
        for batch in loader:
            x, y, pad_mask, loss_mask = batch
            x_d = x.to(device)
            y_d = y.to(device)
            pad_mask_d = pad_mask.to(device)
            loss_mask_d = loss_mask.to(device)

            if accepts_attention_mask:
                outputs = model(
                    x_d, targets=y_d, attention_mask=pad_mask_d, get_logits=True
                )
            else:
                outputs = model(x_d, targets=y_d, get_logits=True)

            logits = outputs["logits"]
            metrics = _counting_metrics_from_outputs(logits, y_d, loss_mask_d)
            # Accumulate (weighted by the valid-position count per batch
            # so the final mean is exact; exact-match is per-sample).
            B, T, V = logits.shape
            valid_total = float(loss_mask_d.sum().item())
            running["n_samples"] += B
            running["per_position_correct"] += metrics["per_position_accuracy"] * valid_total
            running["per_position_total"] += valid_total
            running["exact_match_correct"] += metrics["exact_match_accuracy"] * B
            running["exact_match_total"] += B
            running["ce_sum"] += metrics["cross_entropy"] * valid_total
            running["ce_total"] += valid_total

    pp_denom = max(running["per_position_total"], 1.0)
    em_denom = max(running["exact_match_total"], 1)
    ce_denom = max(running["ce_total"], 1.0)
    return CountingRunMetrics(
        n_samples=int(running["n_samples"]),
        exact_match_accuracy=float(running["exact_match_correct"] / em_denom),
        per_position_accuracy=float(running["per_position_correct"] / pp_denom),
        cross_entropy=float(running["ce_sum"] / ce_denom),
    )


def analyse_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    """Load a checkpoint, run baseline + per-repeat ablations, return results."""
    import torch  # noqa: F401
    from torch.utils.data import DataLoader
    from analysis.common.loader import load_model_from_checkpoint
    from data.counting import CountingDataset, TE200_MAX_OUT

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    model, config = load_model_from_checkpoint(
        checkpoint_dir=args.checkpoint,
        checkpoint_file=args.checkpoint_file,
        config_mode=args.config_mode,
        device=device,
        module_path=args.module_path,
    )
    model.eval()
    n_repeat = int(getattr(model, "n_repeat", 1) or 1)
    model_name = getattr(config, "model", "unknown")
    accepts_attention_mask = model_name == "but_full_depth"

    dataset = CountingDataset(
        split="ood",
        seed=int(args.seed),
        num_samples=int(args.n_samples),
        sequence_length=int(args.sequence_length),
        max_out=TE200_MAX_OUT,
        length_range=OOD_LENGTH_RANGE,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
    )

    baseline = run_one_pass(
        model, loader, device,
        target_repeat=0,
        module_path=args.module_path,
        accepts_attention_mask=accepts_attention_mask,
    )

    per_repeat: dict[int, CountingRunMetrics] = {}
    for r in range(1, n_repeat + 1):
        # Rebuild loader each time: DataLoader with num_workers=0 is
        # re-iterable but we want deterministic ordering per repeat.
        loader_r = DataLoader(
            dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=0,
        )
        ablated = run_one_pass(
            model, loader_r, device,
            target_repeat=r,
            module_path=args.module_path,
            accepts_attention_mask=accepts_attention_mask,
        )
        per_repeat[r] = ablated

    importance: dict[str, float] = {}
    for r, ablated in per_repeat.items():
        importance[f"repeat_{r}"] = float(
            baseline.exact_match_accuracy - ablated.exact_match_accuracy
        )

    return {
        "checkpoint": args.checkpoint,
        "checkpoint_file": args.checkpoint_file,
        "model": model_name,
        "n_repeat": n_repeat,
        "n_samples": baseline.n_samples,
        "accepts_attention_mask": accepts_attention_mask,
        "baseline": dataclasses.asdict(baseline),
        "per_repeat_ablation": {
            f"repeat_{r}": dataclasses.asdict(m) for r, m in per_repeat.items()
        },
        "per_repeat_importance": importance,
        "meta": {
            "ood_length_range": list(OOD_LENGTH_RANGE),
            "sequence_length": int(args.sequence_length),
            "seed": int(args.seed),
            "module_path": args.module_path,
        },
    }


# ---------------------------------------------------------------------------
# Smoke test (pure-numpy synthetic 4-repeat residual pipeline)
# ---------------------------------------------------------------------------


def _synthetic_forward(
    x0: np.ndarray,
    repeat_deltas: list[np.ndarray],
    ablate_repeat: int | None = None,
) -> np.ndarray:
    """Simulate a 4-repeat residual stream with optional identity-ablation.

    Parameters
    ----------
    x0 : ndarray (D,)
        Initial residual.
    repeat_deltas : list[ndarray]
        Per-repeat additive contribution; ``output = x + deltas[r]``
        at repeat ``r + 1`` (1-indexed outer).
    ablate_repeat : int | None
        When not None and in ``[1, len(repeat_deltas)]``, the delta at
        that repeat is replaced with zero (identity-replacement).

    Returns
    -------
    ndarray (D,)
        Final residual after all repeats.
    """
    x = x0.copy()
    for r_idx, delta in enumerate(repeat_deltas, start=1):
        effective_delta = (
            np.zeros_like(delta)
            if ablate_repeat is not None and r_idx == ablate_repeat
            else delta
        )
        x = x + effective_delta
    return x


def _smoke_test() -> None:
    """Pure-numpy DV-4 smoke test with known arithmetic."""
    n_repeat = 4
    D = 3
    x0 = np.zeros(D, dtype=np.float64)
    # Per-repeat deltas: unit direction per repeat index so baseline = sum of units.
    repeat_deltas = [
        np.array([1.0, 0.0, 0.0]),  # repeat 1
        np.array([0.0, 1.0, 0.0]),  # repeat 2
        np.array([0.0, 0.0, 1.0]),  # repeat 3
        np.array([0.0, 0.0, 0.0]),  # repeat 4 is a "zero-contribution" repeat
    ]
    target = np.array([1.0, 1.0, 1.0])

    baseline = _synthetic_forward(x0, repeat_deltas)
    baseline_correct = int(np.allclose(baseline, target))
    assert baseline_correct == 1, f"baseline=={baseline} does not equal target=={target}"

    # Ablate each repeat individually; repeats 1-3 should flip the prediction,
    # repeat 4 (zero contribution) should leave the prediction correct.
    importance = {}
    for r in range(1, n_repeat + 1):
        ablated = _synthetic_forward(x0, repeat_deltas, ablate_repeat=r)
        ablated_correct = int(np.allclose(ablated, target))
        importance[r] = baseline_correct - ablated_correct

    assert importance[1] == 1, f"Ablating repeat 1 should flip accuracy; got {importance[1]}"
    assert importance[2] == 1, f"Ablating repeat 2 should flip accuracy; got {importance[2]}"
    assert importance[3] == 1, f"Ablating repeat 3 should flip accuracy; got {importance[3]}"
    assert importance[4] == 0, f"Ablating zero-contribution repeat 4 should NOT flip accuracy; got {importance[4]}"

    print("DV-4 smoke test PASS")
    print(f"  Baseline accuracy (exact match): {baseline_correct}")
    print(f"  Per-repeat importance: {importance}")
    print(
        f"  Non-zero importance for at least one r: "
        f"{any(v != 0 for v in importance.values())}"
    )
    print(
        f"  Zero-contribution repeat correctly identified as zero-importance: "
        f"{importance[4] == 0}"
    )


def main(argv: list[str] | None = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)

    if args.smoke_test:
        _smoke_test()
        return

    if not args.checkpoint:
        parser.error("--checkpoint is required when --smoke-test is not set")
    if not args.output_dir:
        parser.error("--output-dir is required when --smoke-test is not set")

    os.makedirs(args.output_dir, exist_ok=True)
    results = analyse_checkpoint(args)

    out_json = os.path.join(args.output_dir, "counting_dv4_causal_results.json")
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)
    print(f"DV-4 results written to {out_json}")


if __name__ == "__main__":
    main()
