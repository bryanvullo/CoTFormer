"""OWT2 validation iterator sized for mechanistic analysis.

Scope
-----
``iterate_owt2_val`` yields ``(x, y)`` pairs from OWT2 validation,
capped at ``total_tokens`` tokens total (default 2048) at
``sequence_length`` per batch. Independent of `eval.py`'s full-corpus
DDP iterator; the analysis sub-stream runs single-process on a single
GPU and needs a deterministic, short iterator.

Falsifiability relevance
------------------------
Every activation-based protocol's inter-batch coefficient-of-variation
gate (`docs/extend-notes.md` §1.6 "Inter-batch robustness") requires
deterministic sampling so that four independent batches can be drawn
and a CV computed; this iterator exposes the seed-per-call interface
that the CV gate depends on.

Ontological purpose
-------------------
Replaces the duplicated ``get_as_batch`` helper present in BOTH
`eval.py:57` and `get_router_weights.py:57`. The module wraps
``data.utils.get_dataset`` + ``data.utils.Dataset`` rather than
duplicating them.

Body deferred to Phase 2.
"""

from __future__ import annotations

from argparse import Namespace
from typing import Iterator

import torch


def iterate_owt2_val(
    data_dir: str,
    config: Namespace,
    seq_length: int,
    batch_size: int,
    total_tokens: int = 2048,
    device: str = "cuda",
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield up to ``total_tokens`` tokens of OWT2 validation data.

    The iterator terminates after ``ceil(total_tokens / (batch_size *
    seq_length))`` batches have been yielded; deterministic per-call
    seeding is resolved from ``config.data_seed``.
    """
    raise NotImplementedError("body in Phase 2")
