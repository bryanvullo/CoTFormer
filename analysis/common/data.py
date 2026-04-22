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

Implementation
--------------
The iterator reads the OWT2 validation memmap via the project's
``data.utils.get_dataset`` dispatcher, then iterates consecutive
``sequence_length``-length windows starting at token ``start_offset``
(defaults to 0). The consecutive-non-overlapping layout matches the
§1.6 "Inter-batch robustness" requirement that 4 independent batches
can be drawn starting at token offset 0 of length 2048 each, totalling
8192 tokens with no overlap.
"""

from __future__ import annotations

from argparse import Namespace
from typing import Iterator

import numpy as np
import torch

from data.utils import get_dataset


def iterate_owt2_val(
    data_dir: str | None,
    config: Namespace,
    seq_length: int,
    batch_size: int,
    total_tokens: int = 2048,
    device: str = "cuda",
    start_offset: int = 0,
    split: str = "val",
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield ``(x, y)`` batches totalling at least ``total_tokens`` tokens.

    Parameters
    ----------
    data_dir : str | None
        Override for ``config.data_dir``. When ``None``, the config's
        existing ``data_dir`` is used. Passing an explicit path is
        convenient for the analysis sub-stream which runs against a
        shared scratch dataset root.
    config : Namespace
        Loaded model config. Required attributes: ``dataset`` (must
        resolve in ``data.utils.PREPARE_GET_DATASET_MAP``),
        ``vocab_size``. Optional: ``sequence_length`` (unused here --
        ``seq_length`` argument governs).
    seq_length : int
        Per-sample sequence length. All yielded ``x`` tensors have
        shape ``(batch_size, seq_length)``; ``y`` is shifted by one
        position.
    batch_size : int
        Per-batch sample count.
    total_tokens : int
        Stop after at least ``total_tokens`` tokens have been yielded.
        The final batch is always full (``batch_size`` samples); the
        iterator terminates after the batch that pushes the cumulative
        count to ``total_tokens`` or more.
    device : str
        Target device string for the yielded tensors. ``"cpu"``
        yields pinned-memory tensors ready for
        ``.to(device, non_blocking=True)`` elsewhere.
    start_offset : int
        Starting token index into the validation memmap. The inter-batch
        robustness gate uses ``start_offset=0`` for the primary measurement;
        secondary independent batches use ``start_offset=total_tokens``,
        ``2*total_tokens``, ``3*total_tokens`` to guarantee
        non-overlapping windows.
    split : str
        ``"train"`` or ``"val"``. Default ``"val"``; the counting task's
        dispatcher accepts both.

    Yields
    ------
    (x, y) : tuple[Tensor, Tensor]
        ``x`` and ``y`` are ``int64`` tensors of shape
        ``(batch_size, seq_length)``. ``y[b, t] = x[b, t+1]``.
    """
    if data_dir is not None:
        config.data_dir = data_dir

    data = get_dataset(config)
    if split not in data:
        raise KeyError(
            f"iterate_owt2_val: split {split!r} not in dataset "
            f"(available: {sorted(data.keys())})"
        )
    stream = data[split]

    n_windows = max(1, (total_tokens + seq_length - 1) // seq_length)
    per_batch_windows = batch_size
    n_batches = (n_windows + per_batch_windows - 1) // per_batch_windows

    for batch_idx in range(n_batches):
        ix = [
            start_offset + (batch_idx * per_batch_windows + within) * seq_length
            for within in range(per_batch_windows)
        ]
        for i in ix:
            if i + seq_length + 1 > len(stream):
                raise ValueError(
                    f"iterate_owt2_val: stream exhausted at offset {i} "
                    f"(len={len(stream)}, seq_length={seq_length})"
                )
        x = torch.stack(
            [torch.from_numpy(stream[i : i + seq_length].astype(np.int64)) for i in ix]
        )
        y = torch.stack(
            [
                torch.from_numpy(stream[i + 1 : i + 1 + seq_length].astype(np.int64))
                for i in ix
            ]
        )
        if device != "cpu":
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        yield x, y
