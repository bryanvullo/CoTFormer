"""Shared infrastructure for `analysis/` protocol scripts.

Seven narrow modules that every protocol under `analysis/` consumes:

- `loader`     -- checkpoint loading with raw-JSON or argparse config modes.
- `sites`      -- `ActivationSite` enum and architecture-agnostic discovery.
- `collector`  -- one-forward-per-checkpoint hook-based activation collector.
- `data`       -- OWT2 validation iterator sized for analysis (2048 tokens).
- `spectral`   -- effective-rank, participation-ratio, KV-CoRE helpers.
- `stats`      -- partial correlation, paired t-test, per-sequence z-score.
- `plotting`   -- figure-setup, site-label, and palette helpers.

Each module is a single responsibility consumed by 3+ protocols; see
`docs/extend-technical.md` §2 for the per-protocol consumer mapping.
"""
