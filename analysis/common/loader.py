"""Checkpoint loader with raw-JSON and argparse config modes.

Scope
-----
Load an `nn.Module` + `argparse.Namespace` pair from a checkpoint
directory, resolving the config either from `summary.json["args"]` as a
plain dict (``config_mode="raw"``) or by round-tripping through the
project argparse (``config_mode="argparse"``) so the loader reuses
every default in `config/base.py` without duplicating them.

The loader is architecture-agnostic: it accepts a ``module_path``
argument that resolves to either the standard transformer block list
(``model.transformer.h``) used by GPT-2-large for the Protocol
D-calibration Tier 1 substrate, or the CoTFormer mid-block list
(``model.transformer.h_mid``) used by the mechanistic-analysis main
sub-stream. Per `docs/extend-notes.md` §1.7 ("Common loader hook for
module-path generality"), ``module_path`` defaults to
``"model.transformer.h_mid"`` to preserve the main-sub-stream default;
the calibration entry point passes ``module_path="model.transformer.h"``
when running Tier 1.

Falsifiability relevance
------------------------
Every protocol (A, A-ext, B, C, D, D-calibration, E, F, G, H, I) depends
on this loader; a correctness bug here invalidates every downstream
H0 test. Therefore the loader is treated as infrastructure: one
module, one entry point, no divergence with `eval.py`'s full-corpus
DDP loader (which is out of scope for analysis because it wraps a
full eval loop).

Ontological purpose
-------------------
Replaces the two divergent loader helpers currently present in the
repo: `analyze_kv_compression.py:44-74` (raw mode only) and
`get_router_weights.py:313-360` (argparse mode only). The shared
loader is the single source of truth for the checkpoint-loading
contract; all Phase 2 protocol bodies consume it.

Implementation contract
-----------------------
The raw-mode path mirrors `analyze_kv_compression.py:44-74` verbatim
on the state-dict-cleaning semantics (strip ``_orig_mod.`` prefix,
drop keys containing ``attn.bias`` or ``wpe``, load with
``strict=False``). The argparse-mode path mirrors
`get_router_weights.py:331-359` on the JSON-to-argparse round-trip
(excluded keys: ``device``, ``dtype``, ``distributed_backend``).
The ``module_path`` kwarg is new in this module and is consumed by
``analysis.common.sites.enumerate_sites``.
"""

from __future__ import annotations

import json
import os
from argparse import Namespace
from typing import Literal

import torch
import torch.nn as nn

import models


_EXCLUDED_ARGPARSE_KEYS = ("device", "dtype", "distributed_backend")


def _build_namespace_raw(summary_args: dict) -> Namespace:
    """Mirror `analyze_kv_compression.py:44-74` Config-shim construction.

    The raw mode loads the JSON dict verbatim into a Namespace without
    applying any argparse defaults; this preserves the exact config the
    checkpoint was trained with (including the historical ``dtype=None``
    path that defaults to float16 autocast).
    """
    ns = Namespace()
    ns.__dict__.update(summary_args)
    return ns


def _build_namespace_argparse(
    summary_args: dict,
    rem_args: list[str] | None,
) -> Namespace:
    """Round-trip the summary-JSON dict through the project argparse.

    Excludes ``device``, ``dtype``, and ``distributed_backend`` from the
    JSON import so the argparse defaults govern (which sets
    ``dtype=torch.bfloat16`` via the `config/base.py:72` default string
    ``"torch.bfloat16"``). Mirrors the isolation semantics used by
    `get_router_weights.py:331-359` for the Figure-5 config-loading
    confounder isolation experiment.
    """
    import argparse as _argparse

    from config import base as config_base

    ns = Namespace()
    for key, value in summary_args.items():
        if key in _EXCLUDED_ARGPARSE_KEYS:
            continue
        setattr(ns, key, value)

    parser = _argparse.ArgumentParser(allow_abbrev=False)
    resolved = config_base.parse_args(parser, rem_args or [], ns)
    return resolved


def _clean_state_dict(raw: dict) -> dict:
    """Strip `_orig_mod.` prefixes and drop `attn.bias`/`wpe` entries.

    The prefix strip is necessary because `torch.compile` wraps modules
    in an `_orig_mod` attribute when a compiled model is saved; loading
    without the strip raises a key-mismatch error. The `attn.bias` drop
    absorbs checkpoints saved before the flash-attention rewrite (the
    causal mask buffer is now registered non-persistently). The `wpe`
    drop absorbs the positional-encoder refactor (the wpe weight was
    previously absolute but is now rotary on all trained checkpoints).

    Lift from `analyze_kv_compression.py:63-69`.
    """
    cleaned = {}
    for key, value in raw.items():
        new_key = key.replace("_orig_mod.", "")
        if "attn.bias" in new_key:
            continue
        if "wpe" in new_key:
            continue
        cleaned[new_key] = value
    return cleaned


def load_model_from_checkpoint(
    checkpoint_dir: str,
    checkpoint_file: str = "ckpt.pt",
    config_mode: Literal["raw", "argparse"] = "raw",
    rem_args: list[str] | None = None,
    device: str = "cpu",
    dtype: torch.dtype | None = None,
    module_path: str = "model.transformer.h_mid",
) -> tuple[nn.Module, Namespace]:
    """Load a checkpoint into an `nn.Module` and its resolved `Namespace`.

    Parameters
    ----------
    checkpoint_dir : str
        Absolute path to the directory containing ``summary.json`` and
        the serialised checkpoint file.
    checkpoint_file : str
        Name of the checkpoint file within ``checkpoint_dir`` (default
        ``"ckpt.pt"``; per-iteration files use ``"ckpt_<n>.pt"``).
    config_mode : {"raw", "argparse"}
        ``"raw"`` reads ``summary.json["args"]`` as a plain dict and
        constructs a ``Namespace`` shim directly. ``"argparse"``
        round-trips the dict through the project argparse in
        `config/base.py` so unspecified fields fall back to the
        project defaults.
    rem_args : list[str] | None
        Command-line overrides applied only under
        ``config_mode="argparse"``. Ignored in ``"raw"`` mode.
    device : str
        Target device string; state-dict is moved after load.
    dtype : torch.dtype | None
        Optional cast applied to the loaded parameters. None preserves
        the checkpoint's native dtype.
    module_path : str
        Dotted path resolved at hook-registration time to the list of
        transformer blocks. Default ``"model.transformer.h_mid"``
        matches the CoTFormer mid-block layout used by the
        mechanistic-analysis main sub-stream;
        ``"model.transformer.h"`` is used by the
        Protocol D-calibration Tier 1 (GPT-2-large) substrate.

    Returns
    -------
    model : nn.Module
        The loaded model in eval mode with ``requires_grad=False`` on
        every parameter (analysis never back-propagates).
    config : argparse.Namespace
        The resolved configuration Namespace. The attribute
        ``_analysis_module_path`` carries ``module_path`` so downstream
        consumers (``enumerate_sites``, ``ActivationCollector``) can
        read it without threading it through their own kwargs.
    """
    summary_path = os.path.join(checkpoint_dir, "summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"load_model_from_checkpoint: no summary.json in {checkpoint_dir}"
        )
    with open(summary_path, "r") as fh:
        summary = json.load(fh)
    if "args" not in summary:
        raise KeyError(
            f"load_model_from_checkpoint: summary.json at {checkpoint_dir} "
            f"has no 'args' key"
        )
    summary_args = summary["args"]

    if config_mode == "raw":
        config = _build_namespace_raw(summary_args)
    elif config_mode == "argparse":
        config = _build_namespace_argparse(summary_args, rem_args)
    else:
        raise ValueError(
            f"load_model_from_checkpoint: config_mode must be 'raw' or "
            f"'argparse' (got {config_mode!r})"
        )

    config._analysis_module_path = module_path

    model = models.make_model_from_args(config)

    ckpt_path = os.path.join(checkpoint_dir, checkpoint_file)
    if os.path.exists(ckpt_path):
        raw_ckpt = torch.load(ckpt_path, map_location="cpu")
        if isinstance(raw_ckpt, dict) and "model" in raw_ckpt:
            state_dict = _clean_state_dict(raw_ckpt["model"])
        else:
            state_dict = _clean_state_dict(raw_ckpt)
        model.load_state_dict(state_dict, strict=False)
    else:
        raise FileNotFoundError(
            f"load_model_from_checkpoint: no {checkpoint_file} in "
            f"{checkpoint_dir}"
        )

    if dtype is not None:
        model = model.to(dtype=dtype)
    model = model.to(device=device)

    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    return model, config
