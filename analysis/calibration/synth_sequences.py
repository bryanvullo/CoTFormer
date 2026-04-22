"""Deterministic Condition-A / Condition-B synthetic sequence generators.

Scope
-----
Two generators produce the sequence substrate for Protocol
D-calibration. Both take an explicit ``seed`` argument; the seed
alone determines the output, so two calls with the same seed produce
bit-identical sequences. This determinism is required for the
pre-registration audit trail -- the calibration ladder's verdict is
only reproducible if the sequences are reproducible.

- **Condition A (Sparse / Single-Target)**: induction-head-style
  ``... q r ... q ?`` sequences with one informative target. Reply
  ``r`` placed at ``p_r = p_q + 1`` immediately after the first
  occurrence of query token ``q``; query token recurs at position
  253; positions 254-255 are filler. Ground-truth attention target
  ``T_A = {p_r}``.
- **Condition B (Broad-Integration / Multi-Target)**: 10 informative
  target positions drawn uniformly from [1, 251] without replacement,
  each carrying a token from a reserved subvocabulary ``V_inf`` of
  size 100. Query marker at position 252; positions 253-255 filler.
  Ground-truth attention target ``T_B = {p_1, ..., p_10}`` with equal
  weights.

Falsifiability relevance
------------------------
The two conditions are designed to produce equivalent attention
entropies for distinct information contents: a model attending
uniformly over 10 informative positions and a model attending
uniformly over 10 random positions will report the same Shannon
entropy but radically different attention-target accuracies. Without
Condition B, the Spearman gate cannot separate "low entropy =
accurate" from "low entropy = accurate AND broad integration
impossible".

Ontological purpose
-------------------
Substrate generation: these functions are not tests themselves, they
produce the material on which the calibration test runs.

Body deferred to Phase 2.
"""

from __future__ import annotations

import torch


def generate_condition_A(
    n: int,
    L: int,
    vocab_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate ``n`` Condition A sparse-induction sequences of length ``L``.

    Returns
    -------
    tokens : torch.Tensor
        Integer tensor of shape ``(n, L)``.
    query_positions : torch.Tensor
        Integer tensor of shape ``(n,)`` with the recurring-query
        position (always ``L - 3`` under the fixed design).
    target_positions : torch.Tensor
        Integer tensor of shape ``(n,)`` with the ground-truth reply
        position ``p_r`` for each sequence.
    """
    raise NotImplementedError("body in Phase 2")


def generate_condition_B(
    n: int,
    L: int,
    vocab_size: int,
    inf_vocab: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate ``n`` Condition B broad-integration sequences of length ``L``.

    Returns
    -------
    tokens : torch.Tensor
        Integer tensor of shape ``(n, L)``.
    query_positions : torch.Tensor
        Integer tensor of shape ``(n,)`` with the broad-query marker
        position (always ``L - 4`` under the fixed design).
    target_positions : torch.Tensor
        Integer tensor of shape ``(n, 10)`` with the 10 ground-truth
        target positions per sequence.
    """
    raise NotImplementedError("body in Phase 2")
