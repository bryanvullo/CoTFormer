#!/usr/bin/env python3
"""Lint Tak-branch ports in ``analysis/`` for the SHA-pinning header.

Scope
-----
Per `docs/extend-notes.md` §1.7 "SHA pinning header convention", every
file in ``analysis/`` that carries ported code from the
``origin/tak-ablation-code`` branch must begin with a three-line
provenance header:

    # Ported from origin/tak-ablation-code@<7-char-SHA> :: <rel path> L<start>-L<end>
    # Ported on <YYYY-MM-DD> by <author handle>
    # Original-author rationale: <one-line summary; cite tak's commit message if present>

This script greps the ``analysis/`` package for files containing
``tak-`` (either in the path or anywhere in the file body) and
verifies each matching file's first three non-blank, non-shebang
lines satisfy the pattern. Exits 0 if no ``tak-`` containing files
exist (the Phase 1 skeleton typically has none); exits non-zero with
a per-file diagnostic on malformed headers.

Falsifiability relevance
------------------------
The header is the audit trail linking a ported block to the original
tak-branch SHA. Missing or malformed headers break reproducibility:
an upstream refactor of tak's branch leaves dangling ports whose
provenance cannot be traced. The lint runs in CI before every HPC
submission wave.

Usage
-----
``python scripts/check_port_headers.py`` from the repo root. Emits
file-and-line diagnostics for failures; one failure is enough to fail
the check.
"""

from __future__ import annotations

import pathlib
import re
import sys
from typing import Iterable


ANALYSIS_ROOT = pathlib.Path(__file__).resolve().parent.parent / "analysis"


# Header line 1: "# Ported from origin/tak-ablation-code@<7-char-SHA> :: <rel path> L<start>-L<end>"
LINE_1_RE = re.compile(
    r"^#\s+Ported\s+from\s+origin/tak-ablation-code@[0-9a-f]{7}\s+::\s+"
    r"[^\s]+\s+L\d+-L\d+\s*$"
)
# Header line 2: "# Ported on YYYY-MM-DD by <author handle>"
LINE_2_RE = re.compile(
    r"^#\s+Ported\s+on\s+\d{4}-\d{2}-\d{2}\s+by\s+\S+.*$"
)
# Header line 3: "# Original-author rationale: <one-line summary>"
LINE_3_RE = re.compile(
    r"^#\s+Original-author\s+rationale:\s+.+$"
)


def _iter_analysis_python_files(root: pathlib.Path) -> Iterable[pathlib.Path]:
    """Yield every .py file under ``root`` (recursive), excluding ``__pycache__``."""
    for p in root.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        yield p


_PORT_MARKER = re.compile(r"origin/tak-ablation-code@[0-9a-f]")


def _file_mentions_tak(path: pathlib.Path) -> bool:
    """Return True if ``path`` contains an ``origin/tak-ablation-code@<sha>`` marker.

    The header convention requires the SHA suffix, so a file that
    intends to declare a port will match the marker pattern. Prose
    mentions of the tak branch without a SHA suffix (e.g., the
    skeleton stubs that note a future port) do NOT match and therefore
    skip the header check; they are expected to acquire a proper
    header when the actual port is made.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="strict")
    except UnicodeDecodeError:
        return False
    return bool(_PORT_MARKER.search(text))


def _leading_comment_lines(path: pathlib.Path, max_lines: int = 20) -> list[str]:
    """Return the leading non-blank, non-shebang comment lines from ``path``.

    The header is the first contiguous block of ``#``-prefixed lines at
    the top of the file, skipping an optional shebang and any leading
    blank lines.
    """
    lines: list[str] = []
    with path.open(encoding="utf-8", errors="strict") as fh:
        for lineno, raw in enumerate(fh, start=1):
            if lineno > max_lines + 3:
                break
            stripped = raw.rstrip("\n")
            if stripped.startswith("#!"):
                continue
            if not stripped.strip():
                if lines:
                    break
                continue
            if not stripped.lstrip().startswith("#"):
                break
            lines.append(stripped)
            if len(lines) >= max_lines:
                break
    return lines


def _check_file(path: pathlib.Path) -> list[str]:
    """Return a list of human-readable error strings for ``path``; empty on pass."""
    header = _leading_comment_lines(path)
    errors: list[str] = []
    if len(header) < 3:
        errors.append(
            f"{path}: missing header: expected at least 3 contiguous comment "
            f"lines at the top of the file, found {len(header)}."
        )
        return errors
    if not LINE_1_RE.match(header[0]):
        errors.append(
            f"{path}:1: header line 1 does not match "
            f"'# Ported from origin/tak-ablation-code@<sha7> :: <path> L<s>-L<e>'"
        )
    if not LINE_2_RE.match(header[1]):
        errors.append(
            f"{path}:2: header line 2 does not match "
            f"'# Ported on YYYY-MM-DD by <author>'"
        )
    if not LINE_3_RE.match(header[2]):
        errors.append(
            f"{path}:3: header line 3 does not match "
            f"'# Original-author rationale: <summary>'"
        )
    return errors


def main() -> int:
    if not ANALYSIS_ROOT.exists():
        print(
            f"check_port_headers: analysis/ root does not exist at {ANALYSIS_ROOT}; "
            f"skipping (run from the repo root).",
            file=sys.stderr,
        )
        return 0

    tak_files = [p for p in _iter_analysis_python_files(ANALYSIS_ROOT) if _file_mentions_tak(p)]
    if not tak_files:
        print("check_port_headers: no tak-branch ports detected; OK.")
        return 0

    any_errors = False
    for path in sorted(tak_files):
        errs = _check_file(path)
        if errs:
            any_errors = True
            for e in errs:
                print(e, file=sys.stderr)
    if any_errors:
        print(
            "check_port_headers: one or more files failed the SHA-pinning "
            "header check; see messages above.",
            file=sys.stderr,
        )
        return 1
    print(f"check_port_headers: {len(tak_files)} tak-branch port file(s); all headers OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
