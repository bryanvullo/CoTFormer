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
(``model.transformer.h_mid``) used by the M2 main analytic
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

Body deferred to Phase 2; see `docs/extend-notes.md` §1.8 sequencing.
"""

from __future__ import annotations

from argparse import Namespace
from typing import Literal

import torch
import torch.nn as nn


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
        matches the CoTFormer mid-block layout used by the M2 main
        analytic sub-stream; ``"model.transformer.h"`` is used by the
        Protocol D-calibration Tier 1 (GPT-2-large) substrate.

    Returns
    -------
    model : nn.Module
        The loaded model in eval mode with ``requires_grad=False`` on
        every parameter (analysis never back-propagates).
    config : argparse.Namespace
        The resolved configuration Namespace.
    """
    raise NotImplementedError("body in Phase 2")
