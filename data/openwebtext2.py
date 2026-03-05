import os
import subprocess
import glob
import shutil
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset


OWT2_DATA_PATH = os.path.join(os.path.dirname(__file__), "datasets/openwebtext2/")
OWT2_SOURCE_URL = "https://huggingface.co/datasets/segyges/OpenWebText2/resolve/main/openwebtext2.jsonl.zst.tar"
tknzr = tiktoken.get_encoding("gpt2")


def prepare_openwebtext2_data(config):
    pass


def get_openwebtext2_data(config):
    """ Downloads (if needed), tokenizes, and caches OpenWebText2 as memmap bin files.

    Source: https://openwebtext2.readthedocs.io/en/latest/
    The original HuggingFace dataset (the_pile_openwebtext2) is defunct.
    Downloads the raw tarball from a HuggingFace mirror of the exact same data.

    Expects: internet access for first run (or a pre-placed tarball / raw files)
    Produces: <data_path>/train.bin, <data_path>/val.bin
    """
    if hasattr(config, 'data_dir') and config.data_dir is not None:
        data_path = os.path.join(config.data_dir, "openwebtext2/")
    else:
        data_path = OWT2_DATA_PATH

    num_proc = 40
    if not os.path.exists(os.path.join(data_path, 'train.bin')):
        os.makedirs(data_path, exist_ok=True)

        raw_dir = os.path.join(data_path, "raw")
        tarball = os.path.join(data_path, "openwebtext2.jsonl.zst.tar")

        # Step 1: Download tarball if raw files don't exist yet
        data_files = sorted(glob.glob(os.path.join(raw_dir, "**/*.jsonl.zst"), recursive=True))
        if not data_files:
            min_tarball_size = 1_000_000  # 1 MB sanity check
            if not os.path.exists(tarball) or os.path.getsize(tarball) < min_tarball_size:
                if os.path.exists(tarball) and os.path.getsize(tarball) < min_tarball_size:
                    os.remove(tarball)
                print(f"Downloading OpenWebText2 from {OWT2_SOURCE_URL} (~28 GB)...")
                subprocess.run(["wget", "-c", OWT2_SOURCE_URL, "-O", tarball], check=True)

            if os.path.getsize(tarball) < min_tarball_size:
                raise RuntimeError(
                    f"Tarball at {tarball} is only {os.path.getsize(tarball)} bytes — "
                    "download likely failed. Delete it and retry, or download locally "
                    "and rsync to this path."
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

        # Step 2: Load raw data and split
        print(f"Loading {len(data_files)} raw files...")
        dataset = load_dataset("json", data_files=data_files)

        split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
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

    train_data = np.memmap(os.path.join(data_path, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_path, 'val.bin'), dtype=np.uint16, mode='r')

    return {'train': train_data, 'val': val_data}
