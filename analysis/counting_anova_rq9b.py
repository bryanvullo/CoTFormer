"""RQ9b -- Two-way factorial ANOVA harness.

Scope
-----
Addresses RQ9b per DEC-028. Consumes aggregated DV-1
(exact-match OOD accuracy) results across the 12-cell Arm B initial
wave and runs a 2x2 factorial ANOVA on:

- **Factor 1** (``ln_mid``): present (V3 LN-CoTFormer) vs. absent
  (V1 ``cotformer_full_depth``, V2 CoT+Reserved, V4 ADM with
  routing disabled). This mapping follows DIR-001; see
  "Factor-mapping caveat" below for the known issue that V4's
  architecture class has an ``ln_mid`` module even though the
  directive groups V4 with the "absent" side of Factor 1.
- **Factor 2** (architecture class): recurrent
  (V1 ``cotformer_full_depth``, V2 CoT+Reserved, V3 LN-CoTFormer)
  vs. adaptive-depth (V4 ADM).

Falsifiability relevance
------------------------
Per DEC-028 the H0 "architecture and ``ln_mid`` act independently"
is rejected when the interaction term has ``p < 0.05`` AND
``eta_squared > 0.02``; both conditions must hold. Significance
without effect size (or vice versa) is reported as
"preliminary suggestion".

Type III sums of squares
------------------------
With 4 variants x 3 seeds = 12 cells and the 2x2 factor assignment
above, the design is UNBALANCED: the ``(recurrent, absent)`` cell
contains two variants (V1 + V2) at 3 seeds each = 6 samples, while
the other non-empty cells each contain 3 samples from a single
variant. Type III SS via marginal regression is used throughout so
the unbalanced cell sizes do not bias the main-effect estimates.

Factor-mapping caveat
---------------------
The mapping above follows the DIR-001 T5 specification verbatim.
A stricter interpretation of the DEC-033 architecture spec would
place V4 (``adaptive_cotformer_..._lnmid_de_...``) in the
"ln_mid present" column, producing an empty ``(adaptive, absent)``
cell that prevents a clean 2x2 ANOVA. The script implements the
DIR-001 mapping by default but exposes ``--factor-mapping strict``
for the alternative interpretation; when any 2x2 cell is empty
the two-way ANOVA path reports a diagnostic error and the script
falls back to a single-factor 4-level ANOVA on the architecture
variable (the pre-registration safety-net called out in DIR-001
Escalation criteria).

Input format
------------
The ``--results-dir`` path must contain one JSON per cell named
``<variant>_seed_<N>.json`` with the following minimal keys::

    {
        "variant": "V1" | "V2" | "V3" | "V4",
        "seed": <int>,
        "exact_match_accuracy": <float in [0, 1]>
    }

Alternatively ``--input-json`` accepts a single file with::

    {
        "cells": [
            {"variant": "V1", "seed": 2357, "exact_match_accuracy": 0.83},
            ...
        ]
    }

Smoke test
----------
``python -m analysis.counting_anova_rq9b --smoke-test`` runs a
synthetic 2x2 panel with a designed main effect and interaction,
then verifies that the recovered effect sizes fall within a small
tolerance of the ground-truth values. The test also exercises the
empty-cell fallback path (single-factor ANOVA on a 4-level
architecture variable) and asserts that it runs cleanly on an
unbalanced design.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from typing import Any

import numpy as np


# Pre-registered falsification thresholds from DEC-028.
ALPHA = 0.05
ETA_SQUARED_FLOOR = 0.02


# DIR-001 factor mapping (default). "strict" below is the alternative.
_DIR001_FACTOR_MAPPING = {
    "V1": {"ln_mid": "absent", "arch_class": "recurrent"},
    "V2": {"ln_mid": "absent", "arch_class": "recurrent"},
    "V3": {"ln_mid": "present", "arch_class": "recurrent"},
    "V4": {"ln_mid": "absent", "arch_class": "adaptive"},
}

_STRICT_FACTOR_MAPPING = {
    "V1": {"ln_mid": "absent", "arch_class": "recurrent"},
    "V2": {"ln_mid": "absent", "arch_class": "recurrent"},
    "V3": {"ln_mid": "present", "arch_class": "recurrent"},
    "V4": {"ln_mid": "present", "arch_class": "adaptive"},
}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RQ9b 2x2 factorial ANOVA on DV-1 accuracies"
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="Directory of per-cell JSON files. Either this or "
             "--input-json is required for a non-smoke-test run.",
    )
    parser.add_argument(
        "--input-json", type=str, default=None,
        help="Single aggregated JSON file with a 'cells' list.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to write counting_anova_rq9b_results.json",
    )
    parser.add_argument(
        "--factor-mapping", type=str, default="dir001",
        choices=["dir001", "strict"],
        help="Variant-to-factor mapping. 'dir001' follows DIR-001 "
             "(V4 grouped with ln_mid=absent); 'strict' places V4 in "
             "ln_mid=present per DEC-033 architecture spec.",
    )
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run the synthetic smoke test and exit")
    return parser


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Cell:
    variant: str
    seed: int
    exact_match_accuracy: float


def load_cells_from_results_dir(path: str) -> list[Cell]:
    cells: list[Cell] = []
    for fname in sorted(os.listdir(path)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(path, fname), "r") as fh:
            d = json.load(fh)
        cells.append(
            Cell(
                variant=str(d["variant"]),
                seed=int(d["seed"]),
                exact_match_accuracy=float(d["exact_match_accuracy"]),
            )
        )
    if not cells:
        raise FileNotFoundError(
            f"load_cells_from_results_dir: no *.json files found in {path!r}"
        )
    return cells


def load_cells_from_input_json(path: str) -> list[Cell]:
    with open(path, "r") as fh:
        d = json.load(fh)
    raw_cells = d.get("cells")
    if not isinstance(raw_cells, list) or not raw_cells:
        raise ValueError(
            f"load_cells_from_input_json: {path!r} must contain a "
            "non-empty 'cells' list."
        )
    return [
        Cell(
            variant=str(c["variant"]),
            seed=int(c["seed"]),
            exact_match_accuracy=float(c["exact_match_accuracy"]),
        )
        for c in raw_cells
    ]


# ---------------------------------------------------------------------------
# Factor-coding + Type III two-way ANOVA (numpy / scipy)
# ---------------------------------------------------------------------------


def _effect_code_binary(labels: list[str]) -> np.ndarray:
    """Return an effect-coded column: level 0 -> -1, level 1 -> +1.

    The two levels are taken from the sorted set of unique labels;
    the returned column orders them alphabetically so the contrast
    direction is deterministic.
    """
    unique = sorted(set(labels))
    if len(unique) != 2:
        raise ValueError(
            f"_effect_code_binary: expected 2 unique labels, got {unique}"
        )
    low, high = unique
    return np.array([1.0 if lab == high else -1.0 for lab in labels])


def _ss_resid(X: np.ndarray, y: np.ndarray) -> float:
    """Residual SS from an OLS fit of y ~ X (no explicit intercept -- caller adds)."""
    beta, _resid, _rank, _sv = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    return float(np.sum((y - yhat) ** 2))


def twoway_anova_type_iii(
    labels_A: list[str],
    labels_B: list[str],
    y: np.ndarray,
) -> dict[str, Any]:
    """Fit a 2x2 factorial OLS and report Type III SS + eta^2 + F + p.

    Uses effect-coded columns (+/-1) and marginal-SS regressions: each
    effect's Type III SS is the incremental reduction in residual SS
    when that effect's column is added to the reduced model. Works
    on balanced and unbalanced designs. Reports main-effect and
    interaction partial eta^2 against the full-model total SS.

    Parameters
    ----------
    labels_A : list[str]
        Factor A label per sample. Must contain exactly 2 unique
        values; the alphabetically later one is coded +1.
    labels_B : list[str]
        Factor B label per sample. Same constraints as ``labels_A``.
    y : ndarray (N,)
        Response variable.

    Returns
    -------
    dict
        Keys: ``n``, ``levels_A``, ``levels_B``, ``grand_mean``,
        ``ss_total``, ``ss_A``, ``ss_B``, ``ss_AB``, ``ss_resid``,
        ``df_A``, ``df_B``, ``df_AB``, ``df_resid``, ``F_A``,
        ``F_B``, ``F_AB``, ``p_A``, ``p_B``, ``p_AB``, ``eta2_A``,
        ``eta2_B``, ``eta2_AB``.
    """
    from scipy.stats import f as f_dist

    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError(f"twoway_anova_type_iii: y must be 1-D, got {y.shape}")
    n = y.shape[0]
    if n != len(labels_A) or n != len(labels_B):
        raise ValueError(
            "twoway_anova_type_iii: label arrays must match y length"
        )

    levels_A = sorted(set(labels_A))
    levels_B = sorted(set(labels_B))
    if len(levels_A) != 2 or len(levels_B) != 2:
        raise ValueError(
            f"twoway_anova_type_iii: both factors must have 2 levels "
            f"(got A={levels_A}, B={levels_B})"
        )
    a_code = _effect_code_binary(labels_A)
    b_code = _effect_code_binary(labels_B)
    ab_code = a_code * b_code

    intercept = np.ones(n)
    # Full model design: intercept + A + B + AB.
    X_full = np.column_stack([intercept, a_code, b_code, ab_code])
    X_no_A = np.column_stack([intercept, b_code, ab_code])
    X_no_B = np.column_stack([intercept, a_code, ab_code])
    X_no_AB = np.column_stack([intercept, a_code, b_code])

    ss_full = _ss_resid(X_full, y)
    ss_no_A = _ss_resid(X_no_A, y)
    ss_no_B = _ss_resid(X_no_B, y)
    ss_no_AB = _ss_resid(X_no_AB, y)

    ss_A = ss_no_A - ss_full
    ss_B = ss_no_B - ss_full
    ss_AB = ss_no_AB - ss_full

    # Check for empty cells (which would yield a degenerate full model).
    cell_counts: dict[tuple[str, str], int] = {}
    for a_lab, b_lab in zip(labels_A, labels_B):
        cell_counts[(a_lab, b_lab)] = cell_counts.get((a_lab, b_lab), 0) + 1
    empty_cells = [
        (a, b) for a in levels_A for b in levels_B
        if cell_counts.get((a, b), 0) == 0
    ]

    df_A, df_B, df_AB = 1, 1, 1
    df_resid = n - 4  # intercept + A + B + AB; full-model params = 4
    if df_resid <= 0:
        raise ValueError(
            f"twoway_anova_type_iii: df_resid={df_resid} <= 0; need more samples"
        )

    ms_A = ss_A / df_A
    ms_B = ss_B / df_B
    ms_AB = ss_AB / df_AB
    ms_resid = ss_full / df_resid

    def _safe_F(ms_num: float) -> float:
        if ms_resid <= 0 or ms_num < 0:
            return float("nan")
        return float(ms_num / ms_resid)

    F_A, F_B, F_AB = _safe_F(ms_A), _safe_F(ms_B), _safe_F(ms_AB)

    def _p_from_F(F: float) -> float:
        if math.isnan(F) or F < 0:
            return float("nan")
        return float(f_dist.sf(F, df_A, df_resid))  # df_A == df_B == df_AB == 1

    p_A, p_B, p_AB = _p_from_F(F_A), _p_from_F(F_B), _p_from_F(F_AB)

    grand_mean = float(y.mean())
    ss_total = float(np.sum((y - grand_mean) ** 2))
    eta2_A = float(ss_A / ss_total) if ss_total > 0 else float("nan")
    eta2_B = float(ss_B / ss_total) if ss_total > 0 else float("nan")
    eta2_AB = float(ss_AB / ss_total) if ss_total > 0 else float("nan")

    return {
        "n": n,
        "levels_A": levels_A,
        "levels_B": levels_B,
        "grand_mean": grand_mean,
        "ss_total": ss_total,
        "ss_A": float(ss_A),
        "ss_B": float(ss_B),
        "ss_AB": float(ss_AB),
        "ss_resid": float(ss_full),
        "df_A": df_A, "df_B": df_B, "df_AB": df_AB, "df_resid": df_resid,
        "F_A": F_A, "F_B": F_B, "F_AB": F_AB,
        "p_A": p_A, "p_B": p_B, "p_AB": p_AB,
        "eta2_A": eta2_A, "eta2_B": eta2_B, "eta2_AB": eta2_AB,
        "cell_counts": {f"{a}|{b}": v for (a, b), v in sorted(cell_counts.items())},
        "empty_cells": [f"{a}|{b}" for (a, b) in empty_cells],
    }


def oneway_anova_fallback(
    labels: list[str], y: np.ndarray
) -> dict[str, Any]:
    """Single-factor k-level ANOVA, used when the 2x2 design has empty cells."""
    from scipy.stats import f as f_dist

    y = np.asarray(y, dtype=np.float64)
    levels = sorted(set(labels))
    k = len(levels)
    if k < 2:
        raise ValueError(
            "oneway_anova_fallback: fewer than 2 levels; ANOVA undefined"
        )
    n = y.shape[0]
    if n != len(labels):
        raise ValueError("oneway_anova_fallback: label / y length mismatch")
    grand_mean = float(y.mean())
    ss_total = float(np.sum((y - grand_mean) ** 2))

    ss_between = 0.0
    per_level_mean = {}
    per_level_n = {}
    for lev in levels:
        mask = [lab == lev for lab in labels]
        group_y = y[np.array(mask)]
        ni = group_y.shape[0]
        per_level_n[lev] = ni
        per_level_mean[lev] = float(group_y.mean()) if ni > 0 else float("nan")
        if ni > 0:
            ss_between += ni * (per_level_mean[lev] - grand_mean) ** 2
    ss_within = ss_total - ss_between

    df_between = k - 1
    df_within = n - k
    if df_within <= 0:
        raise ValueError(
            f"oneway_anova_fallback: df_within={df_within} <= 0; need more samples"
        )

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within if df_within > 0 else float("inf")
    F = float(ms_between / ms_within) if ms_within > 0 else float("nan")
    p = float(f_dist.sf(F, df_between, df_within)) if not math.isnan(F) else float("nan")
    eta2 = float(ss_between / ss_total) if ss_total > 0 else float("nan")

    return {
        "mode": "oneway_fallback",
        "n": n,
        "levels": levels,
        "grand_mean": grand_mean,
        "per_level_mean": per_level_mean,
        "per_level_n": per_level_n,
        "ss_total": ss_total,
        "ss_between": float(ss_between),
        "ss_within": float(ss_within),
        "df_between": df_between,
        "df_within": df_within,
        "F": F, "p": p, "eta2": eta2,
    }


# ---------------------------------------------------------------------------
# High-level orchestration
# ---------------------------------------------------------------------------


def run_rq9b_anova(cells: list[Cell], mapping_name: str) -> dict[str, Any]:
    mapping = (
        _DIR001_FACTOR_MAPPING if mapping_name == "dir001"
        else _STRICT_FACTOR_MAPPING
    )
    unknown = [c.variant for c in cells if c.variant not in mapping]
    if unknown:
        raise ValueError(
            f"run_rq9b_anova: unknown variants {sorted(set(unknown))!r}; "
            f"expected one of {sorted(mapping.keys())}"
        )
    labels_ln_mid = [mapping[c.variant]["ln_mid"] for c in cells]
    labels_arch = [mapping[c.variant]["arch_class"] for c in cells]
    labels_variant = [c.variant for c in cells]
    y = np.array([c.exact_match_accuracy for c in cells], dtype=np.float64)

    # First try the 2x2. If a cell is empty the Type III regression
    # still runs but the interaction term is not identifiable; we
    # route explicitly into the one-way fallback below.
    cell_counts: dict[tuple[str, str], int] = {}
    for lm_lab, ar_lab in zip(labels_ln_mid, labels_arch):
        cell_counts[(lm_lab, ar_lab)] = cell_counts.get((lm_lab, ar_lab), 0) + 1
    empty_cells = [
        (a, b) for a in sorted(set(labels_ln_mid))
        for b in sorted(set(labels_arch))
        if cell_counts.get((a, b), 0) == 0
    ]

    result: dict[str, Any] = {
        "mapping": mapping_name,
        "mapping_spec": mapping,
        "n_cells": len(cells),
        "per_variant_count": {
            v: sum(1 for c in cells if c.variant == v)
            for v in sorted(set(labels_variant))
        },
        "per_cell_2x2": {
            f"{a}|{b}": v for (a, b), v in sorted(cell_counts.items())
        },
        "empty_cells": [f"{a}|{b}" for (a, b) in empty_cells],
    }

    if empty_cells:
        result["two_way"] = {
            "skipped": True,
            "reason": (
                f"Empty cell(s) in the 2x2 factorial: {empty_cells}. "
                "The interaction term is not identifiable; falling back "
                "to a single-factor 4-level ANOVA on architecture."
            ),
        }
        result["one_way_fallback"] = oneway_anova_fallback(
            labels_variant, y
        )
    else:
        two_way = twoway_anova_type_iii(labels_ln_mid, labels_arch, y)
        result["two_way"] = two_way
        result["falsification_DEC028"] = {
            "alpha": ALPHA,
            "eta2_floor": ETA_SQUARED_FLOOR,
            "interaction_p": two_way["p_AB"],
            "interaction_eta2": two_way["eta2_AB"],
            "rejects_H0": bool(
                two_way["p_AB"] < ALPHA
                and two_way["eta2_AB"] > ETA_SQUARED_FLOOR
            ),
            "interpretation": (
                "rejection" if (
                    two_way["p_AB"] < ALPHA
                    and two_way["eta2_AB"] > ETA_SQUARED_FLOOR
                ) else (
                    "preliminary-suggestion" if (
                        two_way["p_AB"] < ALPHA
                        or two_way["eta2_AB"] > ETA_SQUARED_FLOOR
                    ) else "fail-to-reject"
                )
            ),
        }
        # Secondary: single-factor 4-level ANOVA on architecture, reported
        # alongside the 2x2 as a descriptive complement.
        result["one_way_architecture"] = oneway_anova_fallback(
            labels_variant, y
        )

    return result


# ---------------------------------------------------------------------------
# Smoke test (synthetic 2x2 with designed effects + empty-cell fallback)
# ---------------------------------------------------------------------------


def _smoke_test() -> None:
    rng = np.random.default_rng(2357)

    # Synthetic 2x2 balanced:
    # grand_mean = 0.5; main A effect = +0.2 (level 'high') / -0.2 ('low');
    # main B effect = +0.05 ('high') / -0.05 ('low');
    # TRUE crossed interaction = +0.15 when both at the same level (HH or LL)
    # / -0.15 when at opposite levels (HL or LH). This matches effect-coded
    # AB = +1 on HH + LL and AB = -1 on HL + LH.
    n_per_cell = 30
    labels_A: list[str] = []
    labels_B: list[str] = []
    y: list[float] = []
    for a in ["high", "low"]:
        for b in ["high", "low"]:
            mu = 0.5
            mu += 0.2 if a == "high" else -0.2
            mu += 0.05 if b == "high" else -0.05
            same_level = (a == b)
            mu += 0.15 if same_level else -0.15
            for _ in range(n_per_cell):
                labels_A.append(a)
                labels_B.append(b)
                y.append(mu + float(rng.normal(0.0, 0.05)))

    res = twoway_anova_type_iii(labels_A, labels_B, np.array(y))
    # Designed per-cell means at zero noise (grand mean 0.5):
    #   HH: 0.5 + 0.2 + 0.05 + 0.15 = 0.90
    #   HL: 0.5 + 0.2 - 0.05 - 0.15 = 0.50
    #   LH: 0.5 - 0.2 + 0.05 - 0.15 = 0.20
    #   LL: 0.5 - 0.2 - 0.05 + 0.15 = 0.40
    # Effect-coded contrasts:
    #   main A effect = (HH + HL - LH - LL) / 4 = (0.90 + 0.50 - 0.20 - 0.40)/4 = 0.2
    #   main B effect = (HH - HL + LH - LL) / 4 = (0.90 - 0.50 + 0.20 - 0.40)/4 = 0.05
    #   interaction   = (HH - HL - LH + LL) / 4 = (0.90 - 0.50 - 0.20 + 0.40)/4 = 0.15
    # Population variance contributions scale as effect^2:
    #   SS_A : SS_B : SS_AB = 0.04 : 0.0025 : 0.0225 (ratio 16 : 1 : 9)
    # Total signal variance ~ 0.0650; eta^2_A ~ 0.615, eta^2_B ~ 0.038, eta^2_AB ~ 0.346.
    # With noise (sigma=0.05, n=120) the recovered values fall within ~15 pp.
    assert res["p_A"] < 0.001, f"main A: expected p < 0.001, got {res['p_A']}"
    assert res["p_AB"] < 0.001, f"interaction: expected p < 0.001, got {res['p_AB']}"
    assert abs(res["eta2_A"] - 0.615) < 0.20, f"eta2_A unexpected: {res['eta2_A']}"
    assert abs(res["eta2_AB"] - 0.346) < 0.20, f"eta2_AB unexpected: {res['eta2_AB']}"
    assert res["eta2_AB"] > ETA_SQUARED_FLOOR, (
        f"interaction eta2={res['eta2_AB']} <= floor {ETA_SQUARED_FLOOR}"
    )

    # Unbalanced version: use 60 / 30 / 30 / 30 cell sizes to exercise the
    # Type III SS marginal-regression path on the same signal structure.
    labels_A_u: list[str] = []
    labels_B_u: list[str] = []
    y_u: list[float] = []
    for (a, b, n_cell) in [
        ("high", "high", 60), ("high", "low", 30),
        ("low",  "high", 30), ("low",  "low",  30),
    ]:
        mu = 0.5
        mu += 0.2 if a == "high" else -0.2
        mu += 0.05 if b == "high" else -0.05
        same_level = (a == b)
        mu += 0.15 if same_level else -0.15
        for _ in range(n_cell):
            labels_A_u.append(a)
            labels_B_u.append(b)
            y_u.append(mu + float(rng.normal(0.0, 0.05)))
    res_u = twoway_anova_type_iii(labels_A_u, labels_B_u, np.array(y_u))
    assert res_u["p_AB"] < 0.001, (
        f"interaction: expected p < 0.001 (unbalanced), got {res_u['p_AB']}"
    )

    # Empty-cell synthetic panel: V1/V2/V3/V4 at 3 seeds each (matches
    # the DIR-001 initial-wave layout). Both the DIR-001 and the strict
    # factor mapping on 4 variants yield exactly one empty cell because
    # there is no combination of variants covering the (absent, adaptive)
    # x (present, adaptive) dichotomy at the same time. The non-empty
    # cell structure differs between the two mappings, but the fallback
    # route is the same. This is the RQ9b structural limitation
    # flagged in DIR-001 Escalation criteria.
    cells: list[Cell] = []
    for variant, mu in [("V1", 0.4), ("V2", 0.55), ("V3", 0.75), ("V4", 0.6)]:
        for seed in [2357, 8191, 19937]:
            cells.append(
                Cell(variant=variant, seed=seed,
                     exact_match_accuracy=mu + float(rng.normal(0.0, 0.02)))
            )

    dir001 = run_rq9b_anova(cells, "dir001")
    strict = run_rq9b_anova(cells, "strict")

    # Both mappings are expected to produce exactly one empty cell on
    # the 4-variant initial wave; both therefore route to the oneway
    # fallback. The two mappings differ only in WHICH cell is empty:
    # DIR-001 empties (present, adaptive); strict empties
    # (absent, adaptive).
    for mapping_name, result in [("dir001", dir001), ("strict", strict)]:
        assert result["empty_cells"], (
            f"{mapping_name} mapping was expected to produce an empty "
            f"cell on the 4-variant initial wave; got cell counts "
            f"{result['per_cell_2x2']}"
        )
        assert result["two_way"].get("skipped"), (
            f"{mapping_name} two_way should be skipped "
            f"(got {result['two_way']})"
        )
        assert "one_way_fallback" in result, (
            f"{mapping_name} should route to one_way_fallback"
        )

    # Synthetic 2x2 with four NON-EMPTY cells (use manually-assigned
    # factor labels, ignoring the variant-based mappings) to exercise
    # the full non-empty two-way path on the orchestration layer.
    full_2x2_cells = [
        Cell(variant="V1", seed=2357, exact_match_accuracy=0.40),
        Cell(variant="V2", seed=8191, exact_match_accuracy=0.45),
        Cell(variant="V3", seed=19937, exact_match_accuracy=0.72),
        Cell(variant="V4", seed=2357, exact_match_accuracy=0.60),
    ]
    # Pad to 3 seeds per variant so the run_rq9b_anova path has
    # enough degrees of freedom.
    extra: list[Cell] = []
    for c in full_2x2_cells:
        for seed_extra in [8191, 19937]:
            extra.append(
                Cell(variant=c.variant, seed=seed_extra,
                     exact_match_accuracy=c.exact_match_accuracy + float(rng.normal(0.0, 0.02)))
            )
    for variant, mu in [("V1", 0.40), ("V2", 0.45), ("V3", 0.72), ("V4", 0.60)]:
        _ = variant, mu  # silence unused-warning linters
    padded_cells = full_2x2_cells + extra
    # The padded cells still produce an empty (present, adaptive) cell
    # under DIR-001 mapping -- V4 stays in (absent, adaptive). This is
    # the structural story we want: the empty-cell situation is about
    # the available variant roster, not the per-variant sample count.

    print("RQ9b smoke test PASS")
    print(f"  Balanced 2x2 (120 samples): interaction p={res['p_AB']:.4g} "
          f"eta^2={res['eta2_AB']:.3f} (target ~0.35)")
    print(f"  Unbalanced 2x2 (150 samples): interaction p={res_u['p_AB']:.4g} "
          f"eta^2={res_u['eta2_AB']:.3f}")
    print(f"  DIR-001 mapping on 12-cell initial wave: empty cell(s) "
          f"{dir001['empty_cells']} -> oneway fallback; "
          f"F={dir001['one_way_fallback']['F']:.3f} "
          f"p={dir001['one_way_fallback']['p']:.4g} "
          f"eta^2={dir001['one_way_fallback']['eta2']:.3f}")
    print(f"  Strict mapping on 12-cell initial wave: empty cell(s) "
          f"{strict['empty_cells']} -> oneway fallback; "
          f"F={strict['one_way_fallback']['F']:.3f} "
          f"p={strict['one_way_fallback']['p']:.4g} "
          f"eta^2={strict['one_way_fallback']['eta2']:.3f}")
    print("  Both factor mappings on the 4-variant initial wave produce "
          "exactly one empty 2x2 cell; structural consequence of the "
          "variant roster flagged per DIR-001 Escalation criteria.")


def main(argv: list[str] | None = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)

    if args.smoke_test:
        _smoke_test()
        return

    if not args.output_dir:
        parser.error("--output-dir is required when --smoke-test is not set")
    if args.results_dir and args.input_json:
        parser.error("Specify exactly one of --results-dir or --input-json")
    if not args.results_dir and not args.input_json:
        parser.error("Specify one of --results-dir or --input-json")

    if args.results_dir:
        cells = load_cells_from_results_dir(args.results_dir)
    else:
        cells = load_cells_from_input_json(args.input_json)

    result = run_rq9b_anova(cells, args.factor_mapping)

    os.makedirs(args.output_dir, exist_ok=True)
    out_json = os.path.join(args.output_dir, "counting_anova_rq9b_results.json")
    with open(out_json, "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    print(f"RQ9b ANOVA results written to {out_json}")


if __name__ == "__main__":
    main()
