# Reproducibility Notes

> Running log of issues, decisions, and findings encountered while reproducing
> [CoTFormer (Mohtashami et al., ICLR 2025)][cotformer-paper] for COMP 6258.
> See [`plan.md`](plan.md) for the full implementation roadmap.

## Table of Contents

- [1. Dataset Acquisition](#1-dataset-acquisition)
- [References](#references)

---

## 1. Dataset Acquisition

**Status:** Resolved (March 2025)

### The problem

The original CoTFormer codebase loads OpenWebText2 (OWT2) via
`load_dataset("the_pile_openwebtext2")`, which calls a HuggingFace-hosted
copy of the dataset. This endpoint now returns a `DefunctDatasetError` --
the dataset has been permanently disabled.

### Why OWT2 is hard to find in 2025-26

OWT2 (~28 GB compressed, ~65 GB raw, 17.1M documents) was originally
distributed as part of EleutherAI's *The Pile* [1]. Several factors have
made standalone access increasingly difficult:

1. **Hosting attrition.** Independent mirrors (notably `mystic.the-eye.eu`
   and `eaidata.bmk.sh`) went offline as hosting costs accumulated.
   Both endpoints now timeout or return 404.
2. **HuggingFace content flags.** Multiple HF copies have been flagged or
   disabled due to PII/malware concerns in raw web scrapes and copyright
   takedown requests (e.g., from the AO3 community).
3. **Absorption into The Pile.** Since OWT2 is a component of the 800 GB
   Pile, most maintainers stopped hosting it as a standalone download.

### Why exact data matters

Training on even a slightly different version of OWT2 (e.g., a cleaned
re-scrape or a different split seed) would invalidate direct perplexity
comparisons against Table 1 of the paper. The original codebase uses
`train_test_split(test_size=0.0005, seed=2357)` on the raw data, so we
need the *exact* same document set as input.

### Our solution

We located an exact mirror of the original tarball on HuggingFace:
[`segyges/OpenWebText2`][segyges-mirror] (29,344,276,480 bytes, MIT license).
This contains the identical `openwebtext2.jsonl.zst.tar` that was
previously hosted on the-eye.eu.

The download, extraction, tokenization, and cleanup pipeline is
self-contained in `data/openwebtext2.py`. On Iridis X, it runs via:

```bash
cd ~/CoTFormer && bash iridis/get_dataset/job.sh
```

If the login node cannot reach `huggingface.co`, the tarball can be
downloaded locally and rsync'd -- see [README > Dataset Setup](../README.md#dataset-setup).

**Output:** `train.bin` (~8 GB) and `val.bin` (~4 MB) as uint16 memmap
files under `/scratch/$USER/datasets/openwebtext2/`.

### Alternatives considered and rejected

| Source | Reason rejected |
|--------|-----------------|
| `the_pile_openwebtext2` (HF) | `DefunctDatasetError` -- permanently disabled |
| `mystic.the-eye.eu` | Domain offline since late 2024 |
| `eaidata.bmk.sh` | Domain offline |
| `Geralt-Targaryen/openwebtext2` (HF) | Parquet re-export; not guaranteed identical to original jsonl.zst |
| EleutherAI/the_pile (full) | 800 GB download to extract one ~28 GB component is impractical on HPC |
| European Language Grid | Requires account creation; availability unverified |

---

## References

[cotformer-paper]: https://openreview.net/forum?id=7igPXQFupX
[segyges-mirror]: https://huggingface.co/datasets/segyges/OpenWebText2

1. Gao, L. et al. (2020). *The Pile: An 800GB Dataset of Diverse Text for Language Modeling.* [arXiv:2101.00027](https://arxiv.org/abs/2101.00027)
2. Mohtashami, A. et al. (2025). *CoTFormer: A Chain-of-Thought Driven Architecture with Budget-Adaptive Computation Cost at Inference.* [ICLR 2025](https://openreview.net/forum?id=7igPXQFupX)
3. DeepSeek-AI (2024). *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model.* [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)
