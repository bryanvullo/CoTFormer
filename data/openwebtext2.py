import os
import io
import subprocess
import glob
import shutil
from tqdm import tqdm
import numpy as np
import zstandard as zstd

# orjson (Rust) is 2-6x faster than stdlib json for loads().
# Falls back gracefully — output is identical either way.
try:
    from orjson import loads as json_loads
except ImportError:
    from json import loads as json_loads


OWT2_DATA_PATH = os.path.join(os.path.dirname(__file__), "datasets/openwebtext2/")
OWT2_SOURCE_URL = "https://huggingface.co/datasets/segyges/OpenWebText2/resolve/main/openwebtext2.jsonl.zst.tar"
MIN_TARBALL_SIZE = 1_000_000  # 1 MB sanity check


def prepare_openwebtext2_data(config):
    pass


def cache_tiktoken_bpe() -> None:
    """Pre-cache the GPT-2 BPE vocabulary for offline use on compute nodes.

    tiktoken downloads vocab files on first call to get_encoding().
    This must run on a node with internet access so the cache is warm
    before any offline job imports this module.
    """
    import tiktoken
    print("Caching tiktoken GPT-2 BPE vocabulary...")
    tiktoken.get_encoding("gpt2")
    print("BPE vocabulary cached.")


def download_openwebtext2(data_path: str) -> None:
    """Download the OpenWebText2 tarball from HuggingFace mirror.

    Requires internet access — run on a login node.
    Resumes partial downloads via wget -c.
    Cleanup of stale state is handled by the calling job.sh script.
    """
    os.makedirs(data_path, exist_ok=True)
    tarball = os.path.join(data_path, "openwebtext2.jsonl.zst.tar")

    if os.path.exists(tarball) and os.path.getsize(tarball) >= MIN_TARBALL_SIZE:
        print(f"Tarball already exists at {tarball} ({os.path.getsize(tarball)} bytes), skipping download.")
        return

    # Remove truncated/corrupt tarball before retrying
    if os.path.exists(tarball) and os.path.getsize(tarball) < MIN_TARBALL_SIZE:
        os.remove(tarball)

    print(f"Downloading OpenWebText2 from {OWT2_SOURCE_URL} (~28 GB)...")
    subprocess.run(["wget", "-c", OWT2_SOURCE_URL, "-O", tarball], check=True)

    if os.path.getsize(tarball) < MIN_TARBALL_SIZE:
        raise RuntimeError(
            f"Tarball at {tarball} is only {os.path.getsize(tarball)} bytes — "
            "download likely failed. Delete it and retry, or download locally "
            "and rsync to this path."
        )

    print(f"Download complete: {tarball} ({os.path.getsize(tarball)} bytes)")


def extract_and_tokenize_openwebtext2(data_path: str) -> None:
    """Extract raw .jsonl.zst files, tokenize with tiktoken, write memmap bins.

    Single-pass pipeline: decompress → parse JSON → tokenize → accumulate.
    No HF datasets or Arrow — just tiktoken + numpy memmap.

    Does not require internet — run on a compute node.
    Requires tiktoken BPE cache to be pre-warmed (see cache_tiktoken_bpe).
    Cleans up raw files and tarball after writing bins.
    """
    if os.path.exists(os.path.join(data_path, 'train.bin')):
        print("train.bin already exists, skipping extraction and tokenization.")
        return

    import tiktoken
    tknzr = tiktoken.get_encoding("gpt2")
    eot = tknzr.eot_token  # end-of-text token (50256), appended after each doc
    num_threads = os.cpu_count() or 1

    tarball = os.path.join(data_path, "openwebtext2.jsonl.zst.tar")
    raw_dir = os.path.join(data_path, "raw")

    # --- Step 1: Extract tarball ---
    # Validates extraction completeness by comparing file count against tarball.
    # Re-extracts from scratch if the raw dir is missing or incomplete (e.g. prior
    # crash mid-extraction). Safe to retry without manual cleanup.
    import tarfile

    def _needs_extraction() -> bool:
        existing = glob.glob(os.path.join(raw_dir, "**/*.jsonl.zst"), recursive=True)
        if not existing:
            return True
        # No tarball to compare against — trust whatever files we have
        if not os.path.exists(tarball) or os.path.getsize(tarball) < MIN_TARBALL_SIZE:
            return False
        with tarfile.open(tarball) as tf:
            expected = sum(1 for m in tf.getmembers() if m.name.endswith(".jsonl.zst"))
        if len(existing) < expected:
            print(f"Incomplete extraction: {len(existing)}/{expected} files. Re-extracting...")
            return True
        return False

    if _needs_extraction():
        if not os.path.exists(tarball) or os.path.getsize(tarball) < MIN_TARBALL_SIZE:
            raise FileNotFoundError(
                f"Tarball not found or too small at {tarball}. "
                "Run iridis/download-dataset/job.sh on a login node first."
            )

        # Wipe partial raw dir before re-extracting
        if os.path.exists(raw_dir):
            shutil.rmtree(raw_dir)

        print(f"Extracting to {raw_dir}...")
        os.makedirs(raw_dir, exist_ok=True)
        with tarfile.open(tarball) as tf:
            tf.extractall(raw_dir)

    data_files = sorted(glob.glob(os.path.join(raw_dir, "**/*.jsonl.zst"), recursive=True))
    if not data_files:
        raise FileNotFoundError(
            f"No .jsonl.zst files found in {raw_dir} after extraction. "
            "Check the tarball contents."
        )

    # --- Step 2: Single-pass tokenize ---
    # Each file -> decompress -> JSON parse -> batch tokenize -> numpy uint16 array.
    # We store one numpy array per file (179 arrays, ~8 GB total) instead of one
    # growing array.array, which doubles its allocation and caused OOM at 36 GB.
    # NOTE: see docs/reprod_notes.md for train/val split divergence.
    file_arrays = []
    doc_lengths = []  # tokens per doc (incl. EOT) — needed for split partitioning

    # encode_ordinary_batch uses Rust threads (num_threads) for parallel BPE encoding
    encode_batch = getattr(tknzr, 'encode_ordinary_batch', None)

    dctx = zstd.ZstdDecompressor()
    for fpath in tqdm(data_files, desc="tokenizing"):
        # Decompress + parse: stream zstd → lines → JSON → extract "text" field
        file_texts = []
        with open(fpath, "rb") as fh:
            reader = dctx.stream_reader(fh)
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            for line in text_stream:
                file_texts.append(json_loads(line)["text"])

        # Tokenize all docs in this file as one batch (Rust-parallel)
        if encode_batch is not None:
            ids_batch = encode_batch(file_texts, num_threads=num_threads)
        else:
            ids_batch = [tknzr.encode_ordinary(t) for t in file_texts]
        del file_texts

        # Flatten all doc tokens into one list, tracking per-doc lengths.
        # One np.array() call per file avoids 3.4M individual allocations.
        file_all_ids = []
        for ids in ids_batch:
            ids.append(eot)
            doc_lengths.append(len(ids))
            file_all_ids.extend(ids)
        file_arrays.append(np.array(file_all_ids, dtype=np.uint16))
        del ids_batch, file_all_ids

    # --- Step 3: Generate train/val split ---
    # Deterministic split using numpy RNG. We now know n_docs from the single pass.
    # The split differs from the original paper's HF train_test_split() — see
    # docs/reprod_notes.md §1 for why this is unavoidable and negligible.
    n_docs = len(doc_lengths)
    n_val = max(1, round(n_docs * 0.0005))
    rng = np.random.default_rng(2357)
    perm = rng.permutation(n_docs)
    val_set = set(perm[:n_val].tolist())
    print(f"Split: {n_docs} docs -> {n_docs - n_val} train, {n_val} val")

    # --- Step 4: Partition tokens into train/val ---
    # Concatenate per-file arrays into one contiguous buffer, then slice by doc
    # boundaries. Since val is only ~0.05% (~1700 docs), we iterate ~1700 gaps
    # to build contiguous train segments rather than looping over 3.4M docs.
    all_np = np.concatenate(file_arrays)
    del file_arrays
    offsets = np.concatenate([[0], np.cumsum(doc_lengths, dtype=np.int64)])
    del doc_lengths

    # Val: gather ~1700 doc slices (numpy views, no copy until concatenate)
    val_sorted = sorted(val_set)
    val_np = np.concatenate([all_np[offsets[i]:offsets[i + 1]] for i in val_sorted])

    # Train: collect contiguous segments between val doc gaps
    train_segments = []
    prev_end = 0
    for vi in val_sorted:
        if offsets[vi] > prev_end:
            train_segments.append(all_np[prev_end:offsets[vi]])
        prev_end = int(offsets[vi + 1])
    if prev_end < len(all_np):
        train_segments.append(all_np[prev_end:])
    train_np = np.concatenate(train_segments)
    del all_np, offsets, train_segments

    # --- Step 5: Write memmap bin files ---
    # uint16 memmap: training code reads these with .astype(np.int64) → torch tensor.
    # Max token ID is 50256 (EOT) which fits in uint16 (max 65535).
    for split, tokens_np in [('train', train_np), ('val', val_np)]:
        filename = os.path.join(data_path, f'{split}.bin')
        n = len(tokens_np)
        arr = np.memmap(filename, dtype=np.uint16, mode='w+', shape=(n,))
        arr[:] = tokens_np
        arr.flush()
        print(f"Wrote {filename}: {n:,} tokens ({n * 2 / 1e9:.2f} GB)")

    del train_np, val_np

    # --- Step 6: Cleanup ---
    # Remove raw .jsonl.zst files and tarball (~94 GB freed).
    print("Cleaning up raw files and tarball...")
    shutil.rmtree(raw_dir, ignore_errors=True)
    if os.path.exists(tarball):
        os.remove(tarball)

    print("Done. Wrote train.bin and val.bin.")


def get_openwebtext2_data(config):
    """Load pre-built train/val memmap bins for training.

    Bins must already exist — run download-dataset and extract-tokenize-dataset
    jobs first (see iridis/ packages).
    """
    if hasattr(config, 'data_dir') and config.data_dir is not None:
        data_path = os.path.join(config.data_dir, "openwebtext2/")
    else:
        data_path = OWT2_DATA_PATH

    train_bin = os.path.join(data_path, 'train.bin')
    val_bin = os.path.join(data_path, 'val.bin')

    if not os.path.exists(train_bin) or not os.path.exists(val_bin):
        raise FileNotFoundError(
            f"Missing {train_bin} or {val_bin}. "
            "Run iridis/download-dataset/job.sh then bash iridis/extract-tokenize-dataset/job.sh first."
        )

    train_data = np.memmap(train_bin, dtype=np.uint16, mode='r')
    val_data = np.memmap(val_bin, dtype=np.uint16, mode='r')

    return {'train': train_data, 'val': val_data}