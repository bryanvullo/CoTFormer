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
The collector maintains a per-forward repeat counter via two
cooperating hooks:

- A forward-pre-hook on the model (``model`` itself) resets the
  counter to 1 at the start of every forward pass.
- A forward-post-hook on ``model.transformer.ln_mid`` increments
  the counter at the end of every repeat; the hook fires even when
  ``ln_mid`` is ``nn.Identity`` (C1 pre-`ln_mid` variant on
  ``cotformer_full_depth_lnmid_depthemb`` with
  ``disable_ln_mid=True``).

Fallback for checkpoints whose model class does not declare
``ln_mid`` at all (the plain ``cotformer_full_depth`` class): the
collector falls back to a forward-post-hook on the last
``h_mid`` block. Because the h_mid loop visits every mid block in
order per repeat, the last block's post-hook fires exactly once per
repeat and serves as the same counter-increment site.

Every ``h_mid[k]`` hook reads the counter at fire time to label
its capture with ``repeat=1..n_repeat``.

Falsifiability relevance
------------------------
Protocols A, C, D, E, F, G, H all depend on per-repeat bookkeeping.
An off-by-one in the repeat counter would silently swap the RQ1
"monotonic convergence" test between adjacent repeats; the counter
therefore has a smoke-test in Phase 2 that asserts the counter reads
1..n_repeat in order across 1000 forward passes.

Ontological purpose
-------------------
The single largest compute saving in the mechanistic-analysis sub-stream:
one collector run per checkpoint replaces ~N separate forward passes
(one per protocol). See `docs/extend-technical.md` §2 "Shared
collector contract" and the per-package Stage-1/Stage-2 layout in
`iridis/analyze-lncot+adm/job.sh`.

Workspace layout
----------------
Per `docs/extend-technical.md` §8.3 (authoritative; §4.3 diagram is the
older stub and is out of sync per mvp-b DEC-004):

    /scratch/$USER/analysis_workspace/<tag>/
    ├── meta.json
    ├── residual_h_begin.npy                     shape (N_tokens, n_embd)
    ├── residual_mid_l<L>_r<R>.npy               shape (N_tokens, n_embd)
    ├── residual_ln_mid_r<R>.npy                 shape (N_tokens, n_embd)
    ├── residual_h_end.npy                       shape (N_tokens, n_embd)
    ├── residual_pre_ln_f.npy                    shape (N_tokens, n_embd); INPUT to ln_f (canonical Belrose 2023 target residual)
    ├── kv_mid_l<L>_r<R>.npy                     shape (N_tokens, 3*n_embd); Q||K||V
    ├── attn_weights_mid_l<L>_r<R>.npy           shape (N_tokens, n_head, T_k); only when ATTN_WEIGHTS requested (Protocol C)
    └── router_mod<R>.npy                        shape (N_tokens, 1); ROUTER_LOGITS

ATTN_WEIGHTS capture and collector-only principle exception
-----------------------------------------------------------
The ``ATTN_WEIGHTS`` site is the second pre-approved derogation of the
collector-only principle (``docs/extend-technical.md`` §8.5); the first
is the DEC-005 KV capture inside ``analysis/kv_rank.py``. Because
``CausalSelfAttention.forward`` returns only ``y`` (the residual
projection output) and not ``(y, att)``, the collector installs a
monkey-patch on each ``CausalSelfAttention.forward`` that re-runs the
non-flash Q/K math path post-softmax to recover ``att``, stashes it on
the module as ``_analysis_att_stash``, and returns the same ``y`` as
the original forward. A standard forward-post-hook then reads the
stash and appends to the per-repeat buffer. The monkey-patch is scoped
to ``register_hooks`` / ``remove_hooks``; the original ``forward`` is
saved per module and restored on teardown (see
``_install_attn_monkey_patch`` + ``_restore_attn_monkey_patch``).
``non_flash=True`` is mandatory whenever ``ATTN_WEIGHTS`` is in
``sites`` -- Flash Attention does not return per-head attention
weights -- and the constructor raises ``ValueError`` otherwise.
"""

from __future__ import annotations

import json
import os
from contextlib import AbstractContextManager, nullcontext
from typing import Iterable, Iterator

import numpy as np
import torch
import torch.nn as nn

from .sites import ActivationSite, enumerate_sites


_COUNTER_ATTR = "_analysis_repeat_counter"


def _flatten_bt(tensor: torch.Tensor) -> torch.Tensor:
    """Flatten (B, T, D) -> (B*T, D). Preserves the last dim."""
    if tensor.dim() < 2:
        return tensor
    if tensor.dim() == 2:
        return tensor
    last = tensor.shape[-1]
    return tensor.reshape(-1, last)


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
        collector flips ``self.flash = False`` on every
        ``CausalSelfAttention`` in the mid-block list for the
        duration of this run. Restored on ``remove_hooks``.
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
        self.model = model
        self.sites: list[ActivationSite] = list(sites)
        self.non_flash = non_flash
        self.module_path = module_path
        self._handles: list = []
        self._flash_saved: list[tuple[nn.Module, bool]] = []
        self._attn_forward_saved: list[tuple[nn.Module, object]] = []
        self._hooks_registered = False
        self._buffers: dict[str, list[np.ndarray]] = {}
        self._forward_count = 0

        transformer = getattr(model, "transformer", None)
        if transformer is None:
            raise AttributeError(
                "ActivationCollector: model has no `transformer` attribute"
            )
        self._transformer = transformer

        n_repeat = getattr(model, "n_repeat", None)
        if n_repeat is None:
            n_repeat = getattr(getattr(model, "config", None), "n_repeat", 1)
        self.n_repeat = int(n_repeat) if n_repeat else 1

        # ATTN_WEIGHTS capture: Flash Attention does not return per-head
        # attention weights, so the collector MUST run on the non-flash
        # math backend. The monkey-patch installed in ``register_hooks``
        # re-runs the Q/K path to recover ``att`` post-softmax and
        # stashes it on the module; the forward-post-hook reads the
        # stash. See docs/extend-technical.md §8.6 for the rationale.
        self._patch_attn_forward = ActivationSite.ATTN_WEIGHTS in self.sites
        if self._patch_attn_forward and not self.non_flash:
            raise ValueError(
                "ActivationCollector: ATTN_WEIGHTS site requires "
                "non_flash=True. Flash Attention (torch SDPA flash "
                "backend) does not return per-head attention weights; "
                "the collector flips attn.flash=False for the duration "
                "of the run via _apply_non_flash. Set non_flash=True "
                "when requesting ATTN_WEIGHTS. See "
                "docs/extend-technical.md §8.6."
            )

    # --------------------------------------------------------------
    # Hook management
    # --------------------------------------------------------------

    def register_hooks(self) -> None:
        """Attach forward hooks at every resolved (site, module) pair."""
        if self._hooks_registered:
            raise RuntimeError(
                "ActivationCollector.register_hooks: already registered"
            )

        setattr(self.model, _COUNTER_ATTR, 1)

        reset_handle = self.model.register_forward_pre_hook(self._reset_counter_hook)
        self._handles.append(reset_handle)

        counter_site = self._install_counter_increment_hook()

        for site in self.sites:
            pairs = enumerate_sites(self.model, site, module_path=self.module_path)
            for key, module in pairs:
                if site is ActivationSite.RESIDUAL_PRE_LN_F:
                    # Pre-hook captures the INPUT to ln_f (the pre-normalisation
                    # residual), matching the canonical Belrose 2023 Tuned Lens
                    # target-residual convention. See _make_pre_hook_site.
                    handle = module.register_forward_pre_hook(
                        self._make_pre_hook_site(site, key)
                    )
                else:
                    handle = module.register_forward_hook(
                        self._make_site_hook(site, key)
                    )
                self._handles.append(handle)

        if self.non_flash:
            self._apply_non_flash()

        if self._patch_attn_forward:
            # Install the ATTN_WEIGHTS capture monkey-patch AFTER the
            # non-flash toggle so the patched forward can defensively
            # verify attn.flash == False at call time.
            self._install_attn_monkey_patch()

        self._hooks_registered = True
        self._counter_site = counter_site

    def remove_hooks(self) -> None:
        """Release every hook registered by ``register_hooks``."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._restore_attn_monkey_patch()
        self._restore_flash()
        if hasattr(self.model, _COUNTER_ATTR):
            delattr(self.model, _COUNTER_ATTR)
        self._hooks_registered = False

    def __enter__(self) -> "ActivationCollector":
        self.register_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.remove_hooks()

    # --------------------------------------------------------------
    # Per-forward counter bookkeeping
    # --------------------------------------------------------------

    def _reset_counter_hook(self, module, inputs):
        """Forward-pre-hook on ``self.model`` that resets the counter."""
        setattr(self.model, _COUNTER_ATTR, 1)

    def _install_counter_increment_hook(self) -> str:
        """Install the increment hook on ln_mid, or fall back to h_mid[-1].

        Returns a string naming the increment site used, for debugging.
        """
        ln_mid = getattr(self._transformer, "ln_mid", None)
        if ln_mid is not None:
            handle = ln_mid.register_forward_hook(self._counter_increment_hook)
            self._handles.append(handle)
            return "ln_mid"

        # Fallback: C1 cotformer_full_depth class has no ln_mid. Hook
        # on the last h_mid block; it fires exactly once per repeat,
        # at the very end of each mid pass.
        mid_path = self.module_path
        if mid_path.startswith("model."):
            mid_path = mid_path[len("model.") :]
        cursor = self.model
        for name in mid_path.split("."):
            cursor = getattr(cursor, name)
        if not hasattr(cursor, "__len__") or len(cursor) == 0:
            raise RuntimeError(
                "ActivationCollector: cannot install per-repeat counter -- "
                "model has neither `transformer.ln_mid` nor a non-empty "
                f"block list at {self.module_path!r}"
            )
        last_mid = cursor[len(cursor) - 1]
        handle = last_mid.register_forward_hook(self._counter_increment_hook)
        self._handles.append(handle)
        return f"{self.module_path}[-1]"

    def _counter_increment_hook(self, module, inputs, output):
        """Advance the repeat counter at the end of each repeat."""
        current = int(getattr(self.model, _COUNTER_ATTR, 1))
        setattr(self.model, _COUNTER_ATTR, current + 1)

    # --------------------------------------------------------------
    # Per-site capture
    # --------------------------------------------------------------

    def _make_site_hook(self, site: ActivationSite, key: str):
        """Return a forward-post-hook closure for ``site`` at ``key``."""

        def _hook(module, inputs, output):
            if output is None:
                return

            repeat_label = int(getattr(self.model, _COUNTER_ATTR, 1))
            # Clamp to [1, n_repeat] in case the increment-hook has
            # already fired one too many times (which happens at
            # n_repeat's ln_mid post-hook).
            repeat_label = max(1, min(repeat_label, self.n_repeat))

            # ATTN_WEIGHTS: read the per-head attention matrix from the
            # stash set by ``_install_attn_monkey_patch``. Skip the
            # standard ``output`` processing; ``output`` here is y
            # (the residual projection) and not the attention weights.
            if site is ActivationSite.ATTN_WEIGHTS:
                att_stash = getattr(module, "_analysis_att_stash", None)
                if att_stash is None:
                    raise RuntimeError(
                        "ActivationCollector ATTN_WEIGHTS hook: expected "
                        "module attribute `_analysis_att_stash` set by "
                        "the monkey-patched CausalSelfAttention.forward "
                        "(see _install_attn_monkey_patch); got None. "
                        "Did the monkey-patch fail to install?"
                    )
                # att_stash shape: (B, n_head, T_q, T_k); already on CPU
                # in float32 (the monkey-patched forward detaches).
                B, H, Tq, Tk = att_stash.shape
                # Reshape to (B * T_q, n_head, T_k) so concatenation
                # across batches on axis 0 produces
                # (N_tokens, n_head, T_k); one row per (batch, query)
                # position with the full per-head attention distribution
                # over key positions.
                captured_flat = (
                    att_stash.permute(0, 2, 1, 3)
                    .contiguous()
                    .reshape(B * Tq, H, Tk)
                    .numpy()
                )
                group_bare = (
                    key.replace("[", "_l")
                    .replace("]", "")
                    .replace(".attn", "")
                )
                buf_key = f"attn_weights_{group_bare}_r{repeat_label}"
                self._buffers.setdefault(buf_key, []).append(captured_flat)
                # Clear stash to surface missing refresh on the next
                # fire instead of silently reusing a stale tensor.
                module._analysis_att_stash = None
                return

            captured = output.detach().to(dtype=torch.float32, copy=False).cpu()
            captured_flat = _flatten_bt(captured).numpy()

            if site is ActivationSite.RESIDUAL_POST_BEGIN:
                buf_key = "residual_h_begin"
            elif site is ActivationSite.RESIDUAL_POST_MID:
                buf_key = f"residual_{key.replace('[', '_l').replace(']', '')}_r{repeat_label}"
            elif site is ActivationSite.RESIDUAL_POST_LN_MID:
                buf_key = f"residual_ln_mid_r{repeat_label}"
            elif site is ActivationSite.RESIDUAL_POST_END:
                buf_key = "residual_h_end"
            elif site is ActivationSite.RESIDUAL_PRE_LN_F:
                # Post-hook branch unused for this site; pre-hook writes
                # the ``residual_pre_ln_f`` buffer directly. Kept here for
                # defensive completeness in case a future caller mis-wires
                # the hook type.
                buf_key = "residual_pre_ln_f"
            elif site is ActivationSite.KV_C_ATTN:
                buf_key = f"kv_{key.replace('[', '_l').replace(']', '').replace('.attn.c_attn', '')}_r{repeat_label}"
            elif site is ActivationSite.ROUTER_LOGITS:
                buf_key = f"router_{key.replace('[', '').replace(']', '').replace('.mod_router', '')}"
            else:
                buf_key = f"unknown_{site.value}_{key}_r{repeat_label}"

            self._buffers.setdefault(buf_key, []).append(captured_flat)

        return _hook

    def _make_pre_hook_site(self, site: ActivationSite, key: str):
        """Return a forward-pre-hook closure for ``site`` at ``key``.

        The pre-hook receives ``(module, inputs)`` and captures
        ``inputs[0]`` (the tensor fed into the module). Used for
        ``RESIDUAL_PRE_LN_F`` where we need the INPUT to ``ln_f``
        rather than its output, matching the canonical Belrose 2023
        Tuned Lens target-residual convention (the translator is
        trained to predict the pre-``ln_f`` residual, then ``ln_f``
        is applied once downstream before ``lm_head``).
        """

        def _hook(module, inputs):
            if not inputs:
                return
            tensor = inputs[0]
            if tensor is None:
                return
            captured = tensor.detach().to(dtype=torch.float32, copy=False).cpu()
            captured_flat = _flatten_bt(captured).numpy()

            if site is ActivationSite.RESIDUAL_PRE_LN_F:
                buf_key = "residual_pre_ln_f"
            else:
                buf_key = f"pre_hook_{site.value}_{key}"

            self._buffers.setdefault(buf_key, []).append(captured_flat)

        return _hook

    # --------------------------------------------------------------
    # Non-flash toggle (for ATTN_WEIGHTS)
    # --------------------------------------------------------------

    def _apply_non_flash(self) -> None:
        """Flip ``attn.flash = False`` on every CausalSelfAttention.

        Saves the previous value so ``_restore_flash`` can revert. On
        checkpoints that were initialised with ``flash=True`` (the
        default on PyTorch 2.0+), ``CausalSelfAttention.__init__`` never
        registered the ``bias`` causal-mask buffer because the
        flash-kernel branch uses ``is_causal=True`` instead. The
        non-flash math path at ``models/base.py``
        ``CausalSelfAttention.forward`` line 94 uses
        ``self.bias[:,:,:T,:T]``; without a buffer this crashes with
        AttributeError. Install a plain-attribute causal mask on the
        fly when the buffer is missing, keyed by the flag
        ``_analysis_nonflash_bias`` so ``_restore_flash`` can remove
        only collector-installed masks (leaving checkpoint-native
        buffers intact).
        """
        transformer = self._transformer
        for group_name in ("h_begin", "h_mid", "h_end", "h"):
            block_list = getattr(transformer, group_name, None)
            if block_list is None:
                continue
            for block in block_list:
                attn = getattr(block, "attn", None)
                if attn is None:
                    continue
                if not hasattr(attn, "flash"):
                    continue
                self._flash_saved.append((attn, bool(attn.flash)))
                attn.flash = False
                existing = getattr(attn, "bias", None)
                if existing is None:
                    config = getattr(attn, "config", None)
                    if config is None:
                        raise RuntimeError(
                            "ActivationCollector._apply_non_flash: attn "
                            "has no ``bias`` buffer and no ``config`` "
                            "attribute to derive sequence_length from; "
                            "cannot install the causal mask required "
                            "by the non-flash math path"
                        )
                    seq_len = int(getattr(config, "sequence_length"))
                    window = getattr(config, "attention_window_length", None)
                    bias = torch.tril(torch.ones(seq_len, seq_len))
                    if window is not None:
                        bias = torch.triu(bias, diagonal=-int(window))
                    bias = bias.view(1, 1, seq_len, seq_len)
                    first_param = next(attn.parameters(), None)
                    if first_param is not None:
                        bias = bias.to(device=first_param.device)
                    # Plain attribute assignment (not register_buffer)
                    # -- we do NOT want this mask to appear in the
                    # module's state_dict, since that would pollute any
                    # downstream save and diverge from the checkpoint-
                    # native layout. Attribute access still works via
                    # standard Python __dict__ lookup; ``del attr.bias``
                    # in ``_restore_flash`` removes it cleanly.
                    attn.bias = bias
                    attn._analysis_nonflash_bias = True

    def _restore_flash(self) -> None:
        for attn, prev in self._flash_saved:
            attn.flash = prev
            if getattr(attn, "_analysis_nonflash_bias", False):
                try:
                    del attn.bias
                except AttributeError:
                    pass
                try:
                    del attn._analysis_nonflash_bias
                except AttributeError:
                    pass
        self._flash_saved.clear()

    # --------------------------------------------------------------
    # ATTN_WEIGHTS monkey-patch (second collector-only-principle exception)
    # --------------------------------------------------------------

    def _install_attn_monkey_patch(self) -> None:
        """Install the att-stashing monkey-patch on every CausalSelfAttention.

        Because ``CausalSelfAttention.forward`` returns only ``y`` (the
        residual-projected output) and not ``(y, att)``, the collector
        cannot recover ``att`` via a standard forward-post-hook. The
        monkey-patch replaces each ``attn.forward`` with a wrapper that:

        1. Calls the original forward to produce ``y`` (no behavioural
           change to downstream code; the residual stream is identical).
        2. Re-runs the non-flash Q / K math path on the same input to
           recover ``att`` post-softmax (mirrors
           ``models/base.py`` CausalSelfAttention.forward lines 65-103).
        3. Stashes ``att`` on the module as ``_analysis_att_stash``
           (detached, float32, CPU) so the forward-post-hook can read it.

        The re-computation approximately doubles the per-block compute
        cost of the non-flash path (itself ~4x flash); Protocol C's
        compute budget in ``docs/extend-notes.md`` §1.3 documents the
        resulting ~9 min GPU wall time.

        Scope: only blocks whose ``attn`` module exposes the expected
        CausalSelfAttention interface (``c_attn``, ``flash`` flag, and
        the causal mask buffer ``bias``) are patched. Other attention
        variants silently skip patching.

        Restoration is symmetric: ``_restore_attn_monkey_patch`` deletes
        the instance-level ``forward`` override on each saved module,
        allowing ``attn.forward`` to fall back to the class-level
        descriptor.
        """
        # Imports are deferred to the patching site because the rest of
        # the collector is numpy-only at top level; math + F are only
        # needed when ATTN_WEIGHTS is requested.
        import math
        from torch.nn import functional as F

        transformer = self._transformer
        for group_name in ("h_begin", "h_mid", "h_end", "h"):
            block_list = getattr(transformer, group_name, None)
            if block_list is None:
                continue
            for block in block_list:
                attn = getattr(block, "attn", None)
                if attn is None:
                    continue
                # Require the full CausalSelfAttention interface -- the
                # re-computation below reads c_attn, n_embd, n_head, and
                # the causal mask buffer named ``bias``. Other attention
                # classes (e.g., MLA variants added later) are skipped
                # explicitly so the monkey-patch cannot partially fire
                # on an architecture it does not understand.
                if not (
                    hasattr(attn, "c_attn")
                    and hasattr(attn, "flash")
                    and hasattr(attn, "n_embd")
                    and hasattr(attn, "n_head")
                    and hasattr(attn, "bias")
                ):
                    continue

                # Save the original forward as a bound method reference so
                # the wrapper can call it unchanged. PyTorch's
                # ``nn.Module.__call__`` looks up ``self.forward`` fresh
                # every call, so replacing the instance attribute is
                # sufficient -- no class-level patching required.
                original_forward = attn.forward
                self._attn_forward_saved.append((attn, original_forward))

                def _patched_forward(
                    x,
                    pos_emb_closure,
                    cache_context,
                    start_index,
                    _attn=attn,
                    _original_forward=original_forward,
                    _math=math,
                    _F=F,
                ):
                    # Defensive: the non_flash toggle must have set
                    # attn.flash=False before any forward fires. If
                    # flash is somehow still True, the re-computed att
                    # below would be inconsistent with the flash-backend
                    # ``y``. Surface the contradiction immediately
                    # instead of silently reporting wrong weights.
                    if _attn.flash:
                        raise RuntimeError(
                            "ATTN_WEIGHTS monkey-patch: expected "
                            "attn.flash=False via non_flash=True toggle; "
                            "got flash=True. Did the non_flash toggle "
                            "misfire on this module?"
                        )
                    # 1. Run the original forward unchanged to produce y.
                    y = _original_forward(
                        x, pos_emb_closure, cache_context, start_index
                    )
                    # 2. Re-run Q/K math to recover att post-softmax.
                    # Mirrors models/base.py CausalSelfAttention.forward
                    # lines 65-99; intentionally omits the dropout +
                    # att-prefix cache branches since:
                    #  * attn_dropout is an identity in eval mode, which
                    #    is how the collector runs (``self.model.eval()``
                    #    in ``run``);
                    #  * cache_context is None during non-training
                    #    eval-mode forwards on OWT2 validation.
                    # Protocol C is specified on eval-mode forwards per
                    # docs/extend-notes.md §1.3; non-standard inference
                    # paths must escalate before using ATTN_WEIGHTS.
                    B, T, C = x.size()
                    q, k, _v = _attn.c_attn(x).split(_attn.n_embd, dim=2)
                    head_dim = C // _attn.n_head
                    k = k.view(B, T, _attn.n_head, head_dim).transpose(1, 2)
                    q = q.view(B, T, _attn.n_head, head_dim).transpose(1, 2)
                    q = pos_emb_closure.adapt_queries(
                        q, start_index=start_index
                    )
                    k = pos_emb_closure.adapt_keys(k, start_index=start_index)
                    att = (q @ k.transpose(-2, -1)) * (
                        1.0 / _math.sqrt(k.size(-1))
                    )
                    att = pos_emb_closure.adapt_attention_before_softmax(
                        att,
                        start_query_index=start_index,
                        start_key_index=start_index,
                    )
                    att = att.masked_fill(
                        _attn.bias[:, :, :T, :T] == 0, float("-inf")
                    )
                    att = _F.softmax(att, dim=-1)
                    # 3. Stash on the module for the post-hook to read.
                    _attn._analysis_att_stash = att.detach().to(
                        dtype=torch.float32
                    ).cpu()
                    return y

                # Install as instance attribute; nn.Module.__call__
                # reads self.forward at call time so this override
                # shadows the class-level method without any C-extension
                # awareness.
                attn.forward = _patched_forward

    def _restore_attn_monkey_patch(self) -> None:
        """Undo ``_install_attn_monkey_patch`` for every saved module.

        Deletes the instance-level ``forward`` override so the
        class-level descriptor (``CausalSelfAttention.forward``) is
        visible again. Also clears any residual ``_analysis_att_stash``
        attribute to avoid surfacing stale tensors on a subsequent
        register / remove cycle.
        """
        for attn, original_forward in self._attn_forward_saved:
            try:
                del attn.forward
            except AttributeError:
                # Defensive fallback: if the instance attribute was
                # somehow already removed (e.g., double-restore), leave
                # the class-level descriptor alone.
                pass
            # Double-check the restore landed: if someone else (or a
            # future refactor) mutated the class-level method, re-bind
            # the saved original as an instance attribute so the
            # model's downstream use remains consistent.
            if getattr(attn, "forward", None) is not original_forward:
                # original_forward is the PRE-patch bound method; it is
                # already bound to attn, so assigning it as an instance
                # attribute behaves the same as the class-level
                # descriptor for the lifetime of the module.
                attn.forward = original_forward
            if hasattr(attn, "_analysis_att_stash"):
                attn._analysis_att_stash = None
        self._attn_forward_saved.clear()

    # --------------------------------------------------------------
    # Run
    # --------------------------------------------------------------

    def run(
        self,
        data_iter: Iterator[tuple[torch.Tensor, torch.Tensor]],
        type_ctx: AbstractContextManager | None = None,
        max_tokens: int = 2048,
    ) -> int:
        """Run one collection pass over ``data_iter``.

        Parameters
        ----------
        data_iter : Iterator[tuple[Tensor, Tensor]]
            Iterator yielding ``(x, y)`` batches. The collector runs
            ``model(x)`` per batch in ``torch.no_grad()`` under the
            ``type_ctx`` autocast context.
        type_ctx : AbstractContextManager | None
            Autocast context (e.g. ``torch.amp.autocast(device_type="cuda",
            dtype=torch.bfloat16)``). ``None`` uses ``nullcontext``.
        max_tokens : int
            Stop after ``max_tokens`` tokens have been forwarded.
            Counts ``B * T`` per batch.

        Returns
        -------
        total_tokens : int
            Number of tokens actually forwarded (may be slightly more
            than ``max_tokens`` because the inner loop stops after the
            batch that pushes the count across the limit).
        """
        if not self._hooks_registered:
            raise RuntimeError(
                "ActivationCollector.run: register_hooks must be called "
                "(or the collector used as a context manager) before run"
            )
        ctx = type_ctx if type_ctx is not None else nullcontext()

        total_tokens = 0
        self.model.eval()
        with torch.no_grad():
            for x, _ in data_iter:
                with ctx:
                    self.model(x)
                total_tokens += int(x.shape[0]) * int(x.shape[1])
                self._forward_count += 1
                if total_tokens >= max_tokens:
                    break
        return total_tokens

    # --------------------------------------------------------------
    # Save
    # --------------------------------------------------------------

    def save(self, workspace_dir: str, meta_extra: dict | None = None) -> None:
        """Persist buffered captures to ``workspace_dir``.

        Writes one ``.npy`` per site-key (concatenating across batches
        along axis 0) plus a ``meta.json`` with the collector's
        configuration and the per-key shapes and token counts.
        """
        os.makedirs(workspace_dir, exist_ok=True)

        per_key_shape: dict[str, list[int]] = {}
        per_key_n_tokens: dict[str, int] = {}
        for key, chunk_list in self._buffers.items():
            if not chunk_list:
                continue
            concatenated = np.concatenate(chunk_list, axis=0)
            path = os.path.join(workspace_dir, f"{key}.npy")
            np.save(path, concatenated)
            per_key_shape[key] = list(concatenated.shape)
            per_key_n_tokens[key] = int(concatenated.shape[0])

        meta = {
            "module_path": self.module_path,
            "n_repeat": self.n_repeat,
            "non_flash": self.non_flash,
            "sites": [site.value for site in self.sites],
            "forward_count": self._forward_count,
            "counter_site": getattr(self, "_counter_site", "unknown"),
            "per_key_shape": per_key_shape,
            "per_key_n_tokens": per_key_n_tokens,
        }
        if meta_extra:
            meta.update(meta_extra)
        with open(os.path.join(workspace_dir, "meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2, sort_keys=True)

    # --------------------------------------------------------------
    # Inspection
    # --------------------------------------------------------------

    def buffer_keys(self) -> list[str]:
        """Return the sorted list of buffer keys captured so far."""
        return sorted(self._buffers.keys())

    def buffer_for(self, key: str) -> np.ndarray:
        """Return the concatenated buffer for ``key`` (RAM-only, no save)."""
        chunk_list = self._buffers.get(key, [])
        if not chunk_list:
            raise KeyError(f"ActivationCollector.buffer_for: no key {key!r}")
        return np.concatenate(chunk_list, axis=0)
