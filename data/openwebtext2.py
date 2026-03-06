import os
import io
import json
import subprocess
import glob
import shutil
import array as _array
from tqdm import tqdm
import numpy as np
import zstandard as zstd


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

    # Step 1: Extract tarball if raw files don't exist yet
    data_files = sorted(glob.glob(os.path.join(raw_dir, "**/*.jsonl.zst"), recursive=True))
    if not data_files:
        if not os.path.exists(tarball) or os.path.getsize(tarball) < MIN_TARBALL_SIZE:
            raise FileNotFoundError(
                f"Tarball not found or too small at {tarball}. "
                "Run iridis/download-dataset/job.sh on a login node first."
            )

        import tarfile
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

    # Step 2: Count documents (streaming, minimal memory)
    # NOTE: see docs/reprod_notes.md §2 for train/val split divergence.
    print(f"Counting documents across {len(data_files)} files...")
    n_docs = 0
    dctx = zstd.ZstdDecompressor()
    for fpath in tqdm(data_files, desc="counting"):
        with open(fpath, "rb") as fh:
            reader = dctx.stream_reader(fh)
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            for _ in text_stream:
                n_docs += 1

    # Generate train/val split (deterministic, matches test_size=0.0005 seed=2357)
    n_val = max(1, round(n_docs * 0.0005))
    rng = np.random.default_rng(2357)
    perm = rng.permutation(n_docs)
    val_set = set(perm[:n_val].tolist())
    print(f"Split: {n_docs} docs -> {n_docs - n_val} train, {n_val} val")

    # Step 3: Tokenize and accumulate into uint16 arrays (~8 GB for train)
    # Per-file batching leverages tiktoken's internal Rust parallelism.
    train_tokens = _array.array('H')  # uint16, 2 bytes per token
    val_tokens = _array.array('H')
    encode_batch = getattr(tknzr, 'encode_ordinary_batch', None)

    doc_idx = 0
    dctx = zstd.ZstdDecompressor()
    for fpath in tqdm(data_files, desc="tokenizing"):
        file_texts = []
        with open(fpath, "rb") as fh:
            reader = dctx.stream_reader(fh)
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            for line in text_stream:
                file_texts.append(json.loads(line)["text"])

        if encode_batch is not None:
            ids_batch = encode_batch(file_texts, num_threads=num_threads)
        else:
            ids_batch = [tknzr.encode_ordinary(t) for t in file_texts]
        del file_texts

        for ids in ids_batch:
            ids.append(eot)
            target = val_tokens if doc_idx in val_set else train_tokens
            target.extend(_array.array('H', ids))
            doc_idx += 1
        del ids_batch

    # Step 4: Write memmap bin files
    for split, tokens in [('train', train_tokens), ('val', val_tokens)]:
        filename = os.path.join(data_path, f'{split}.bin')
        n = len(tokens)
        arr = np.memmap(filename, dtype=np.uint16, mode='w+', shape=(n,))
        arr[:] = np.frombuffer(tokens, dtype=np.uint16)
        arr.flush()
        print(f"Wrote {filename}: {n:,} tokens ({n * 2 / 1e9:.2f} GB)")

    del train_tokens, val_tokens

    # Step 5: Cleanup raw files and tarball (~94 GB freed)
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
