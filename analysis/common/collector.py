"""`ActivationCollector` -- one forward pass per checkpoint, multi-site hooks.

Scope
-----
A single collector registers forward hooks on every requested site
(see ``analysis.common.sites.ActivationSite``) and runs ONE forward
pass per checkpoint over the shared data iterator in
``analysis.common.data``. Per-site tensors are buffered and written
to a per-checkpoint workspace directory on ``/scratch``; downstream
protocols read from the workspace rather than re-running the
forward.

Per-forward repeat counter
--------------------------
The CoTFormer mid-loop calls the same `h_mid[k]` module ``n_repeat``
times per forward. A naive forward hook cannot distinguish repeats.
The collector maintains a per-forward repeat counter by instrumenting
`ln_mid` (which fires exactly once per repeat, and exists as
``nn.Identity`` in the C1 pre-`ln_mid` variant -- the Identity module
still fires its forward hook, so no special-casing is required).
Each `h_mid[k]` hook reads this counter at fire time to label its
capture with ``repeat=1..n_repeat``.

Falsifiability relevance
------------------------
Protocols A, C, D, E, F, G, H all depend on per-repeat bookkeeping.
An off-by-one in the repeat counter would silently swap the RQ1
"monotonic convergence" test between adjacent repeats; the counter
therefore has a smoke-test in Phase 2 that asserts the counter reads
1..n_repeat in order across 1000 forward passes.

Ontological purpose
-------------------
The single largest compute saving in the M2 main analytic sub-stream:
one collector run per checkpoint replaces ~N separate forward passes
(one per protocol). See `docs/extend-technical.md` §2 "Shared
collector contract" and the per-package Stage-1/Stage-2 layout in
`iridis/analyze-lncot+adm/job.sh`.

Workspace layout
----------------
Per `docs/extend-technical.md` §4.3:

    /scratch/$USER/analysis_workspace/<tag>/
    ├── meta.json
    ├── residual_h_begin.npy
    ├── residual_h_mid_r<R>_l<L>.npy
    ├── residual_ln_mid_r<R>.npy
    ├── kv_h_mid_l<L>.npy          (repeat-averaged; Protocol G weight consumer)
    ├── kv_h_mid_r<R>_l<L>.npy     (per-repeat; Protocol G activation consumer)
    └── attn_weights_h_mid_l<L>.npy (only when ATTN_WEIGHTS site requested)

Body deferred to Phase 2.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Iterable, Iterator

import torch
import torch.nn as nn

from .sites import ActivationSite


class ActivationCollector:
    """Hook-based multi-site activation collector with per-repeat bookkeeping.

    Parameters
    ----------
    model : nn.Module
        Loaded model (typically returned by
        ``analysis.common.loader.load_model_from_checkpoint``).
    sites : Iterable[ActivationSite]
        The set of activation sites to capture on the forward pass.
    non_flash : bool
        Set to ``True`` when any requested site is ``ATTN_WEIGHTS`` --
        Flash Attention does not return attention weights, so the
        collector must monkey-patch the attention path to the non-flash
        branch for the duration of this run.
    module_path : str
        Dotted path to the transformer-block list; see
        ``analysis.common.sites.enumerate_sites`` for the supported
        values.
    """

    def __init__(
        self,
        model: nn.Module,
        sites: Iterable[ActivationSite],
        non_flash: bool = False,
        module_path: str = "model.transformer.h_mid",
    ) -> None:
        raise NotImplementedError("body in Phase 2")

    def register_hooks(self) -> None:
        """Attach forward hooks at every resolved (site, module) pair."""
        raise NotImplementedError("body in Phase 2")

    def run(
        self,
        data_iter: Iterator[tuple[torch.Tensor, torch.Tensor]],
        type_ctx: AbstractContextManager,
        max_tokens: int = 2048,
    ) -> None:
        """Run one forward pass, consuming ``data_iter`` until ``max_tokens``."""
        raise NotImplementedError("body in Phase 2")

    def save(self, workspace_dir: str) -> None:
        """Persist per-site caches + ``meta.json`` to the scratch workspace."""
        raise NotImplementedError("body in Phase 2")

    def remove_hooks(self) -> None:
        """Release every hook registered by ``register_hooks``."""
        raise NotImplementedError("body in Phase 2")
