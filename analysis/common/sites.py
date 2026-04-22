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
- Flat ``h`` layout (C1 pre-`ln_mid` CoTFormer variant; standard
  4L-for-counting baseline; GPT-2-large Tier 1 substrate).
- Split ``h_begin`` / ``h_mid`` / ``h_end`` layout (C2, C3, C4, C5 --
  the M2 main analytic checkpoints).
- `mod[]` router modules (C4, C5 only -- ADM variants).

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

Body deferred to Phase 2.
"""

from __future__ import annotations

from enum import Enum

import torch.nn as nn


class ActivationSite(Enum):
    """Closed vocabulary of activation-capture sites used across protocols."""

    RESIDUAL_POST_BEGIN = "residual_post_begin"
    RESIDUAL_POST_MID = "residual_post_mid"        # per (repeat, mid_layer)
    RESIDUAL_POST_LN_MID = "residual_post_ln_mid"  # per repeat
    RESIDUAL_POST_END = "residual_post_end"
    RESIDUAL_POST_LN_F = "residual_post_ln_f"
    KV_C_ATTN = "kv_c_attn"                        # per mid_layer; fused Q/K/V
    ATTN_WEIGHTS = "attn_weights"                  # per mid_layer; NON-FLASH
    ROUTER_LOGITS = "router_logits"                # per mod[r]; ADM only


def enumerate_sites(
    model: nn.Module,
    site: ActivationSite,
    module_path: str = "model.transformer.h_mid",
) -> list[tuple[str, nn.Module]]:
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
        ``"ln_mid@repeat3"``; the caller uses them as stable
        identifiers in saved caches.
    """
    raise NotImplementedError("body in Phase 2")
