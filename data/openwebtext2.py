import os
import io
import json
import subprocess
import glob
import shutil
from tqdm import tqdm
import numpy as np
import tiktoken
import zstandard as zstd
from datasets import Dataset


OWT2_DATA_PATH = os.path.join(os.path.dirname(__file__), "datasets/openwebtext2/")
OWT2_SOURCE_URL = "https://huggingface.co/datasets/segyges/OpenWebText2/resolve/main/openwebtext2.jsonl.zst.tar"
MIN_TARBALL_SIZE = 1_000_000  # 1 MB sanity check
tknzr = tiktoken.get_encoding("gpt2")


def prepare_openwebtext2_data(config):
    pass


def download_openwebtext2(data_path: str) -> None:
    """Download the OpenWebText2 tarball from HuggingFace mirror.

    Requires internet access — run on a login node.
    Resumes partial downloads via wget -c.
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
    """Extract raw files, stream into Arrow, tokenize, and write memmap bins.

    Does not require internet — run on a compute node.
    Cleans up raw files and tarball after writing bins.
    """
    if os.path.exists(os.path.join(data_path, 'train.bin')):
        print("train.bin already exists, skipping extraction and tokenization.")
        return

    tarball = os.path.join(data_path, "openwebtext2.jsonl.zst.tar")
    raw_dir = os.path.join(data_path, "raw")
    num_proc = min(os.cpu_count() or 1, 16)

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

    # Step 2: Stream raw jsonl.zst files into Arrow via generator (low memory)
    # NOTE: The original codebase used load_dataset("the_pile_openwebtext2") which
    # loaded a pre-built HF Arrow dataset with a fixed row ordering. That dataset is
    # now defunct. We load from raw files instead, which may produce a different row
    # order and therefore a different train/val split (see docs/reprod_notes.md §2).
    # The document *set* is identical; only the partition may differ.
    def _iter_documents(file_list):
        """Yield {"text": ...} dicts from sorted jsonl.zst files."""
        dctx = zstd.ZstdDecompressor()
        for fpath in tqdm(file_list, desc="reading jsonl.zst"):
            with open(fpath, "rb") as fh:
                reader = dctx.stream_reader(fh)
                text_stream = io.TextIOWrapper(reader, encoding="utf-8")
                for line in text_stream:
                    yield {"text": json.loads(line)["text"]}

    print(f"Loading {len(data_files)} raw files (streaming)...")
    dataset = Dataset.from_generator(_iter_documents, gen_kwargs={"file_list": data_files})

    split_dataset = dataset.train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')

    def process(example):
        ids = tknzr.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
        ids.append(tknzr.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
        # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
        out = {'ids': ids, 'len': len(ids)}
        return out

    # Step 3: Tokenize
    tokenized = split_dataset.map(
        process,
        remove_columns=split_dataset['train'].column_names,
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # Step 4: Write memmap bin files
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'])
        filename = os.path.join(data_path, f'{split}.bin')
        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

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
            "Run iridis/download-dataset/job.sh then sbatch iridis/extract-tokenize-dataset/job.sh first."
        )

    train_data = np.memmap(train_bin, dtype=np.uint16, mode='r')
    val_data = np.memmap(val_bin, dtype=np.uint16, mode='r')

    return {'train': train_data, 'val': val_data}
