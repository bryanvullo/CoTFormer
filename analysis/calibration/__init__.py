"""Protocol D-calibration sub-package -- attention-entropy monotone calibration.

Validates the monotone interpretation underlying RQ5 DV-2 (low entropy
implies isolation; high entropy implies broad integration) via an
empirical mapping from attention entropy to attention-target accuracy
on synthetic sequences with known ground-truth targets. See
`docs/extend-notes.md` §1.3 "Protocol D-calibration" for the
three-tier substrate, two-condition sequence design, and four-gate
validation ladder. Frozen thresholds live in ``preregister.md``.
"""
