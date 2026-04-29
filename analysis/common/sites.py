"""`ActivationSite` enum plus architecture-agnostic site-discovery helper.

Scope
-----
Every protocol under `analysis/` registers hooks at a finite set of
named sites. The enum below is the closed vocabulary; the helper
``enumerate_sites`` resolves a site name to a list of
``(human_readable_key, module_reference)`` pairs given a concrete
loaded model. The enum is intentionally closed: a new site requires
an explicit commit that adds a member, enforcing discoverability.

Supported variants
------------------
- Flat ``h`` layout (standard 4L-for-counting baseline; GPT-2-large
  Tier 1 substrate; ``but_full_depth``).
- Split ``h_begin`` / ``h_mid`` / ``h_end`` layout with optional
  ``ln_mid`` (C1, C2, C3, C4, C5 -- every CoTFormer variant). C1
  (``cotformer_full_depth``) does NOT have ``ln_mid`` at the
  ``model.transformer`` level; the collector uses a last-h_mid-block
  fallback to obtain the per-repeat counter there (see
  ``analysis.common.collector.ActivationCollector``).
- `mod[]` router modules (C4, C5 only -- the ADM
  ``adaptive_cotformer_..._single_final`` and the `mod_efficient_*`
  variants; exposed on ``model.transformer.mod``).

Falsifiability relevance
------------------------
RQ1-RQ6 each test a specific H0 about a specific activation site
(e.g. RQ1 requires the post-`h_mid` residual at each repeat; RQ6
requires the post-`ln_mid` residual). The enum enforces that the
protocol-to-site mapping in `docs/extend-notes.md` §1.3 and the code
agree; a typo in a site name surfaces immediately rather than
silently aliasing another site.

Ontological purpose
-------------------
Decouples site identity from module-address details that differ
across CoTFormer variants and across architectures (standard
transformer vs CoTFormer). The protocol code references the enum
member by name; the hook-registration machinery resolves the
concrete module address.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import TYPE_CHECKING, Iterable

# Defer the torch import so the pure-Python helpers
# (``residual_key_prefix``, ``discover_workspace_sites``,
# ``_group_label``, ``site_label``) can be imported in CPU-only
# smoke-test environments without ``torch`` installed -- mirrors the
# deferred-import pattern in ``analysis.counting_dv3_attention`` and
# ``analysis.counting_dv4_causal``. Functions that traverse a live
# ``nn.Module`` (``_mid_block_list``, ``enumerate_sites``) re-import
# ``torch.nn`` inside their bodies; an ImportError there surfaces a
# clear cause if torch is missing AND a model-traversal path is
# invoked. Type hints stay as strings under
# ``from __future__ import annotations``.
if TYPE_CHECKING:
    import torch.nn as nn


class ActivationSite(Enum):
    """Closed vocabulary of activation-capture sites used across protocols."""

    RESIDUAL_POST_BEGIN = "residual_post_begin"
    RESIDUAL_POST_MID = "residual_post_mid"        # per (repeat, mid_layer)
    RESIDUAL_POST_LN_MID = "residual_post_ln_mid"  # per repeat
    RESIDUAL_POST_END = "residual_post_end"
    RESIDUAL_PRE_LN_F = "residual_pre_ln_f"        # input to ln_f; canonical Belrose 2023 target residual
    KV_C_ATTN = "kv_c_attn"                        # per mid_layer; fused Q/K/V
    ATTN_WEIGHTS = "attn_weights"                  # per mid_layer; NON-FLASH
    ROUTER_LOGITS = "router_logits"                # per mod[r]; ADM only


def _resolve_attr_path(root: object, dotted_path: str) -> object:
    """Resolve a dotted attribute path like ``model.transformer.h_mid``.

    Returns the resolved object, or raises ``AttributeError`` with a
    path-contextualised message when any segment is missing.
    """
    parts = dotted_path.split(".")
    cursor = root
    traversed: list[str] = []
    for name in parts:
        if not hasattr(cursor, name):
            raise AttributeError(
                f"resolve_attr_path: attribute {name!r} missing on "
                f"{'.'.join(traversed) or type(root).__name__}; "
                f"full path {dotted_path!r}"
            )
        cursor = getattr(cursor, name)
        traversed.append(name)
    return cursor


def _mid_block_list(model: "nn.Module", module_path: str) -> "nn.ModuleList":
    """Return the list-of-blocks at ``module_path`` (auto-rooted at model).

    The ``module_path`` argument is dotted from the model root
    (`"model.transformer.h_mid"` or `"model.transformer.h"`). If the
    path starts with the literal ``"model."`` prefix (DIR-001 canonical
    form), the prefix is stripped before resolution because the root
    is already the ``nn.Module``.
    """
    import torch.nn as nn  # deferred: model-traversal path requires torch
    path = module_path
    if path.startswith("model."):
        path = path[len("model.") :]
    resolved = _resolve_attr_path(model, path)
    if not isinstance(resolved, (nn.ModuleList, list)):
        raise TypeError(
            f"_mid_block_list: resolved path {module_path!r} is "
            f"{type(resolved).__name__}, expected nn.ModuleList"
        )
    return resolved


def _group_label(module_path: str) -> str:
    """Map a module_path to the human-readable group prefix used in keys.

    ``model.transformer.h_mid`` -> ``mid``; ``model.transformer.h`` ->
    ``flat``. The label is the left-hand side of every
    ``enumerate_sites`` key (e.g. ``"mid[3].attn.c_attn"``).
    """
    leaf = module_path.rsplit(".", 1)[-1]
    if leaf == "h_mid":
        return "mid"
    if leaf == "h":
        return "flat"
    return leaf


def discover_workspace_sites(
    workspace_dir: str, prefix: str
) -> list[tuple[int, int, str]]:
    """Scan ``workspace_dir`` for ``<prefix>_l<L>_r<R>.npy`` capture files.

    Single source of truth for the per-protocol workspace site-discovery
    routine. CKA, effective-dim, and kv-rank all materialise per-(layer,
    repeat) ``.npy`` activations using a common naming convention; this
    helper enumerates them so each protocol does not reimplement the
    listdir + filter + parse + sort dance independently.

    Parameters
    ----------
    workspace_dir : str
        Absolute or relative path to the directory the collector wrote
        ``.npy`` captures into. Must exist; otherwise ``FileNotFoundError``
        is raised (matches ``os.listdir`` semantics).
    prefix : str
        The bare key prefix BEFORE the ``"_l<L>_r<R>"`` suffix --- e.g.
        ``"residual_mid"``, ``"residual_flat"`` (see ``residual_key_prefix``)
        or ``"kv_mid"`` for fused Q||K||V activation captures.

    Returns
    -------
    list[tuple[int, int, str]]
        ``[(layer, repeat, abs_path), ...]`` sorted ascending by
        ``(layer, repeat)``. Files whose stem does not parse as
        ``<prefix>_l<int>_r<int>.npy`` are silently skipped.
    """
    scan_prefix = f"{prefix}_l"
    suffix = ".npy"
    out: list[tuple[int, int, str]] = []
    for name in os.listdir(workspace_dir):
        if not name.startswith(scan_prefix) or not name.endswith(suffix):
            continue
        stem = name[: -len(suffix)]
        body = stem[len(scan_prefix) :]
        if "_r" not in body:
            continue
        layer_str, repeat_str = body.split("_r", 1)
        try:
            layer = int(layer_str)
            repeat = int(repeat_str)
        except ValueError:
            continue
        out.append((layer, repeat, os.path.join(workspace_dir, name)))
    out.sort(key=lambda t: (t[0], t[1]))
    return out


def residual_key_prefix(module_path: str) -> str:
    """Return the group prefix the collector uses for residual buffer keys.

    The collector's _make_site_hook composes buf_key as
    ``f"residual_{group}_l<L>_r<R>"`` where ``<group>`` is the trailing
    module name resolved by ``_group_label``. CoTFormer's ``h_mid``
    yields ``"residual_mid"``; standard ``transformer.h`` (used by
    Protocol D-calibration Tier 1 GPT-2-large substrate) yields
    ``"residual_flat"``.

    This helper is the single source of truth for the prefix string;
    protocols must call it rather than hardcoding ``"residual_mid"``.
    """
    return f"residual_{_group_label(module_path)}"


def enumerate_sites(
    model: "nn.Module",
    site: ActivationSite,
    module_path: str = "model.transformer.h_mid",
) -> list[tuple[str, "nn.Module"]]:
    """Resolve a site name to concrete ``(key, module)`` pairs.

    The ``module_path`` argument selects between the standard
    transformer block list (``"model.transformer.h"``) and the
    CoTFormer mid-block list (``"model.transformer.h_mid"``); see
    ``analysis.common.loader.load_model_from_checkpoint`` for the
    matching default.

    Returns
    -------
    pairs : list[tuple[str, nn.Module]]
        Each element is ``(human_readable_key, module_reference)``.
        Human-readable keys resemble ``"mid[0].attn.c_attn"`` or
        ``"ln_mid"``; the caller uses them as stable identifiers in
        saved caches.
    """
    transformer = _resolve_attr_path(model, "transformer")
    group = _group_label(module_path)
    pairs: list[tuple[str, "nn.Module"]] = []

    if site is ActivationSite.RESIDUAL_POST_BEGIN:
        h_begin = getattr(transformer, "h_begin", None)
        if h_begin is None or len(h_begin) == 0:
            return []
        last = h_begin[len(h_begin) - 1]
        return [("h_begin_last", last)]

    if site is ActivationSite.RESIDUAL_POST_MID:
        block_list = _mid_block_list(model, module_path)
        for layer_idx in range(len(block_list)):
            pairs.append((f"{group}[{layer_idx}]", block_list[layer_idx]))
        return pairs

    if site is ActivationSite.RESIDUAL_POST_LN_MID:
        ln_mid = getattr(transformer, "ln_mid", None)
        if ln_mid is None:
            return []
        return [("ln_mid", ln_mid)]

    if site is ActivationSite.RESIDUAL_POST_END:
        h_end = getattr(transformer, "h_end", None)
        if h_end is None or len(h_end) == 0:
            return []
        last = h_end[len(h_end) - 1]
        return [("h_end_last", last)]

    if site is ActivationSite.RESIDUAL_PRE_LN_F:
        # Canonical Belrose 2023 target residual for the Tuned Lens:
        # the INPUT to ln_f (before normalisation), captured via a
        # forward-pre-hook in ``ActivationCollector``.
        ln_f = getattr(transformer, "ln_f", None)
        if ln_f is None:
            return []
        return [("ln_f", ln_f)]

    if site is ActivationSite.KV_C_ATTN:
        block_list = _mid_block_list(model, module_path)
        for layer_idx in range(len(block_list)):
            block = block_list[layer_idx]
            attn = getattr(block, "attn", None)
            if attn is None:
                continue
            c_attn = getattr(attn, "c_attn", None)
            if c_attn is None:
                continue
            pairs.append((f"{group}[{layer_idx}].attn.c_attn", c_attn))
        return pairs

    if site is ActivationSite.ATTN_WEIGHTS:
        block_list = _mid_block_list(model, module_path)
        for layer_idx in range(len(block_list)):
            block = block_list[layer_idx]
            attn = getattr(block, "attn", None)
            if attn is None:
                continue
            pairs.append((f"{group}[{layer_idx}].attn", attn))
        return pairs

    if site is ActivationSite.ROUTER_LOGITS:
        mod_list = getattr(transformer, "mod", None)
        if mod_list is None or len(mod_list) == 0:
            return []
        for router_idx in range(len(mod_list)):
            mod_block = mod_list[router_idx]
            router = getattr(mod_block, "mod_router", None)
            if router is None:
                continue
            pairs.append((f"mod[{router_idx}].mod_router", router))
        return pairs

    raise ValueError(f"enumerate_sites: unknown site {site!r}")


def sites_per_forward_count(sites: Iterable[ActivationSite], n_repeat: int) -> dict:
    """Return the expected per-forward fire count for each site in ``sites``.

    Used by smoke tests to validate the per-repeat counter bookkeeping
    in ``analysis.common.collector.ActivationCollector``.

    - ``RESIDUAL_POST_BEGIN`` / ``RESIDUAL_POST_END`` / ``RESIDUAL_PRE_LN_F``:
      1 per forward.
    - ``RESIDUAL_POST_MID`` / ``KV_C_ATTN`` / ``ATTN_WEIGHTS``:
      ``n_repeat`` per forward per (layer, repeat) pair.
    - ``RESIDUAL_POST_LN_MID``: ``n_repeat`` per forward.
    - ``ROUTER_LOGITS``: 1 per forward per router (routers are not
      re-used across repeats in the ADM forward).
    """
    fire_counts: dict = {}
    for site in sites:
        if site in {
            ActivationSite.RESIDUAL_POST_BEGIN,
            ActivationSite.RESIDUAL_POST_END,
            ActivationSite.RESIDUAL_PRE_LN_F,
        }:
            fire_counts[site.value] = 1
        elif site in {
            ActivationSite.RESIDUAL_POST_MID,
            ActivationSite.KV_C_ATTN,
            ActivationSite.ATTN_WEIGHTS,
        }:
            fire_counts[site.value] = n_repeat
        elif site is ActivationSite.RESIDUAL_POST_LN_MID:
            fire_counts[site.value] = n_repeat
        elif site is ActivationSite.ROUTER_LOGITS:
            fire_counts[site.value] = 1
        else:
            fire_counts[site.value] = -1
    return fire_counts
