import os
import io
import subprocess
import glob
import shutil
import array as _array
from tqdm import tqdm
import numpy as np
import zstandard as zstd

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
    """Extract raw files, tokenize with tiktoken, and write memmap bins.

    Streams documents from .jsonl.zst files without loading everything into
    memory. No HF datasets or Arrow needed — just tiktoken + numpy memmap.

    Does not require internet — run on a compute node.
    Requires tiktoken BPE cache to be pre-warmed (see cache_tiktoken_bpe).
    Cleans up raw files and tarball after writing bins.
    """
    if os.path.exists(os.path.join(data_path, 'train.bin')):
        print("train.bin already exists, skipping extraction and tokenization.")
        return

    import tiktoken
    tknzr = tiktoken.get_encoding("gpt2")
    eot = tknzr.eot_token
    num_threads = os.cpu_count() or 1

    tarball = os.path.join(data_path, "openwebtext2.jsonl.zst.tar")
    raw_dir = os.path.join(data_path, "raw")

    # Step 1: Extract tarball (re-extracts from scratch if raw dir looks incomplete)
    import tarfile

    def _needs_extraction() -> bool:
        """Check if extraction is needed by comparing file count against tarball."""
        existing = glob.glob(os.path.join(raw_dir, "**/*.jsonl.zst"), recursive=True)
        if not existing:
            return True
        if not os.path.exists(tarball) or os.path.getsize(tarball) < MIN_TARBALL_SIZE:
            return False  # no tarball to compare against — trust existing files
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

        # Wipe partial raw dir to ensure clean extraction
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

    # Step 2: Single-pass tokenize — accumulate all tokens + track doc lengths.
    # This avoids a separate counting pass (which doubled I/O + decompression).
    # NOTE: see docs/reprod_notes.md for train/val split divergence.
    all_tokens = _array.array('H')  # uint16, ~8 GB for full corpus
    doc_lengths = []  # token count per document (including EOT)
    encode_batch = getattr(tknzr, 'encode_ordinary_batch', None)

    dctx = zstd.ZstdDecompressor()
    for fpath in tqdm(data_files, desc="tokenizing"):
        file_texts = []
        with open(fpath, "rb") as fh:
            reader = dctx.stream_reader(fh)
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            for line in text_stream:
                file_texts.append(json_loads(line)["text"])

        if encode_batch is not None:
            ids_batch = encode_batch(file_texts, num_threads=num_threads)
        else:
            ids_batch = [tknzr.encode_ordinary(t) for t in file_texts]
        del file_texts

        # Batch per file: one extend call per file instead of per document.
        file_all_ids = []
        for ids in ids_batch:
            ids.append(eot)
            doc_lengths.append(len(ids))
            file_all_ids.extend(ids)
        all_tokens.extend(_array.array('H', file_all_ids))
        del ids_batch, file_all_ids

    # Step 3: Generate train/val split now that we know n_docs.
    n_docs = len(doc_lengths)
    n_val = max(1, round(n_docs * 0.0005))
    rng = np.random.default_rng(2357)
    perm = rng.permutation(n_docs)
    val_set = set(perm[:n_val].tolist())
    print(f"Split: {n_docs} docs -> {n_docs - n_val} train, {n_val} val")

    # Step 4: Partition tokens into train/val via numpy slicing.
    # Val is ~0.05% of docs (~1700), so we build ~1700 contiguous train
    # segments between val gaps — avoids iterating 3.4M docs in Python.
    all_np = np.frombuffer(all_tokens, dtype=np.uint16)
    offsets = np.concatenate([[0], np.cumsum(doc_lengths, dtype=np.int64)])
    del all_tokens, doc_lengths

    val_sorted = sorted(val_set)
    val_np = np.concatenate([all_np[offsets[i]:offsets[i + 1]] for i in val_sorted])

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

    # Step 5: Write memmap bin files
    for split, tokens_np in [('train', train_np), ('val', val_np)]:
        filename = os.path.join(data_path, f'{split}.bin')
        n = len(tokens_np)
        arr = np.memmap(filename, dtype=np.uint16, mode='w+', shape=(n,))
        arr[:] = tokens_np
        arr.flush()
        print(f"Wrote {filename}: {n:,} tokens ({n * 2 / 1e9:.2f} GB)")

    del train_np, val_np

    # Step 6: Cleanup raw files and tarball (~94 GB freed)
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
