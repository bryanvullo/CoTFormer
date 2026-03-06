# Reproducibility Notes

> Running log of issues, decisions, and findings encountered while reproducing
> [CoTFormer (Mohtashami et al., ICLR 2025)][cotformer-paper] for COMP 6258.
> See [`plan.md`](plan.md) for the full implementation roadmap.

## Table of Contents

- [1. Dataset Acquisition](#1-dataset-acquisition)
- [2. Train/Val Split Divergence](#2-trainval-split-divergence)
- [3. Token Ordering Within Bins](#3-token-ordering-within-bins)
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

The pipeline is split across two Iridis job scripts due to HPC constraints
(login nodes have internet, compute nodes do not):

```bash
cd ~/CoTFormer
bash iridis/download-dataset/job.sh              # login node
bash iridis/extract-tokenize-dataset/job.sh      # compute node (sbatch)
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

## 2. Train/Val Split Divergence

**Status:** Accepted limitation

### The problem

The original codebase calls `load_dataset("the_pile_openwebtext2")`, which
returned a pre-built HuggingFace Arrow dataset with a fixed row ordering.
`train_test_split(test_size=0.0005, seed=2357)` then partitions by integer
row index, so the split is deterministic *only if the row order is
identical*.

Our pipeline diverges from the original in three independent ways, each of
which changes which documents land in train vs. val:

#### 2a. Document ordering

Since the pre-built HF Arrow dataset is permanently defunct (see
[section 1](#1-dataset-acquisition)), we load from the raw `.jsonl.zst`
tarball instead. Files are read in sorted filename order, preserving
document order within each file, but this sequence is not guaranteed to
match the row order of the original pre-built Arrow dataset.

#### 2b. Split RNG implementation

The original code uses HuggingFace datasets' `train_test_split()`, which
internally generates a permutation via numpy. Depending on the library
version installed when the paper's experiments were run, this could be
either `np.random.RandomState(seed)` (legacy, `datasets < 2.x`) or
`np.random.default_rng(seed)` (new Generator API, `datasets >= 2.x`).
These produce entirely different permutation sequences from the same seed.

Our pipeline bypasses HF datasets entirely and generates the split with:

```python
rng = np.random.default_rng(2357)
perm = rng.permutation(n_docs)
val_set = set(perm[:n_val].tolist())
```

Even if we somehow recovered the exact original Arrow row ordering, the
split would still differ unless we matched the exact `datasets` library
version used by the original authors. Since this version is not recorded
in their codebase, faithful reproduction of the split is not possible.

#### 2c. Validation set size calculation

The original uses HF's `test_size=0.0005` parameter, whose internal
rounding (ceiling, floor, or round) determines the exact number of
validation documents. Our code uses `n_val = max(1, round(n_docs * 0.0005))`,
which may differ by one document for certain corpus sizes.

### Why the pre-built dataset is unrecoverable

The pre-built Arrow artifact was hosted exclusively on HuggingFace under
`the_pile_openwebtext2`. It was permanently disabled due to PII/safety
flags and copyright takedown requests. An extensive search (30+ queries
across HuggingFace, EleutherAI, the-eye.eu, ELG, and community mirrors)
found only raw tarball copies -- none preserve the Arrow row ordering.

### Impact assessment

The split ratio is 99.95% train / 0.05% val across ~3.4M documents.
At this ratio, any two random partitions with a fixed seed will produce
statistically equivalent validation sets. We expect negligible impact on
reported perplexity (well within the noise floor of stochastic training).
This is noted as a known deviation in our reproduction.

### HF datasets elimination

Early iterations of the pipeline used HF `datasets` with Arrow as an
intermediary (`load_dataset("json", ...)`, `Dataset.from_dict()`,
`Dataset.from_generator()`). All Arrow-based approaches encountered
prohibitive issues on HPC:

- `load_dataset("json", data_files=...)`: PyArrow `combine_chunks()` crash
  (`ArrowIndexError`) during large JSONL ingestion -- a known bug [3].
- `Dataset.from_dict()`: required holding the full text corpus (~65 GB)
  in both Python strings and an Arrow table simultaneously, causing OOM
  at 210+ GB peak.
- Chunked `pyarrow.array()` per file: still OOM'd due to memory spikes
  during Arrow table consolidation (~80-90 GB sampled, but transient
  peaks exceeded 120-160 GB allocations).
- `Dataset.from_generator()`: low memory but 3x slower due to
  single-threaded disk-backed Arrow cache writes.

The final pipeline eliminates HF `datasets` and PyArrow entirely, using
`zstandard` + `json` for reading, `tiktoken.encode_ordinary_batch()` for
tokenization (Rust-parallel), and `array.array('H')` for accumulation.
Peak memory is ~10-20 GB vs 80-210 GB with Arrow. The `datasets` and
`pyarrow` packages have been removed from `requirements.txt`.

---

## 3. Token Ordering Within Bins

**Status:** Accepted -- no training impact

### The difference

The original pipeline tokenizes each split (train, val) as a separate HF
dataset via `map()`, then writes tokens to memmap in 1024 sharded batches.
Because `train_test_split(shuffle=True)` reindexes the dataset, documents
appear in the bin file in shuffled order.

Our pipeline tokenizes documents in file-encounter order (sorted filenames,
sequential within each file), routing each document to the train or val
accumulator inline. The resulting `train.bin` contains tokens in the
natural file order, not shuffled.

### Why this doesn't matter

The training loop (`optim/base.py`) samples random offsets into the memmap
file for each batch. The global ordering of tokens within `train.bin` has
no effect on training -- every position is equally likely to be sampled
regardless of its physical location in the file. The same applies to
`val.bin` during evaluation.

This is purely a cosmetic difference in on-disk layout.

---

## References

[cotformer-paper]: https://openreview.net/forum?id=7igPXQFupX
[segyges-mirror]: https://huggingface.co/datasets/segyges/OpenWebText2

1. Gao, L. et al. (2020). *The Pile: An 800GB Dataset of Diverse Text for Language Modeling.* [arXiv:2101.00027](https://arxiv.org/abs/2101.00027)
2. Mohtashami, A. et al. (2025). *CoTFormer: A Chain-of-Thought Driven Architecture with Budget-Adaptive Computation Cost at Inference.* [ICLR 2025](https://openreview.net/forum?id=7igPXQFupX)
3. HuggingFace datasets [issue #6206](https://github.com/huggingface/datasets/issues/6206) -- PyArrow offset overflow in `combine_chunks()` during large JSONL ingestion.
4. DeepSeek-AI (2024). *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model.* [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)
