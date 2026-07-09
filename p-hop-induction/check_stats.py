#!/usr/bin/env python
"""Check constructive p-hop data for simple shortcut features.

This is a stdout-friendly version of ``check_stats.ipynb`` for Iridis runs.
It verifies p-hop labels and evaluates cheap majority baselines on held-out
rows, so high-cardinality features do not look predictive just by memorising
the sampled examples.
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    import types

    # generate.py imports torch for its tensor batch helper, but this script
    # only uses token/string generation. A minimal stub keeps local CPU-only
    # environments usable while Iridis will use the real torch module.
    sys.modules["torch"] = types.SimpleNamespace(Tensor=object)

from generate import InductionHopsFinalAnswerTask  # noqa: E402


def parse_task_name(task):
    match = re.fullmatch(r"phop_p(\d+)_seq(\d+)_a(\d+)_final.*", task)
    if match is None:
        raise ValueError(f"Unexpected task name: {task}")
    return {
        "hops": int(match.group(1)),
        "seq_len": int(match.group(2)),
        "char_tokens": int(match.group(3)),
    }


def previous_index(seq, query, before_idx):
    for idx in range(before_idx - 1, -1, -1):
        if seq[idx] == query:
            return idx
    return -1


def compute_path(seq, hops):
    current_idx = len(seq) - 1
    query = seq[current_idx]
    path = [(current_idx, query)]
    for _ in range(hops):
        prev_idx = previous_index(seq, query, current_idx)
        if prev_idx == -1:
            return None, path
        current_idx = prev_idx + 1
        query = seq[current_idx]
        path.append((current_idx, query))
    return query, path


def load_rows_from_file(path, spec, sample_n=None, seed=0, max_rows=None):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for row_idx, line in enumerate(handle):
            if max_rows is not None and row_idx >= max_rows:
                break
            seq, labels = json.loads(line)
            answer = labels[-1]
            computed_answer, path = compute_path(seq, spec["hops"])
            rows.append(
                {
                    "seq": seq,
                    "answer": answer,
                    "path": path,
                    "computed_answer": computed_answer,
                    "valid": computed_answer == answer,
                }
            )
    if sample_n is not None and len(rows) > sample_n:
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(rows), size=sample_n, replace=False)
        rows = [rows[index] for index in indices]
    return rows


def generate_rows(spec, sample_n, seed=0):
    rng = np.random.RandomState(seed)
    task = InductionHopsFinalAnswerTask(
        seq_len=spec["seq_len"],
        char_tokens=spec["char_tokens"],
        min_hops=spec["hops"],
        max_hops=spec["hops"],
        rng=rng,
        ensure_exists=True,
        include_hop_token=False,
        avoid_adjacent_repeats=False,
        sampling_strategy="constructive",
    )
    rows = []
    for _ in range(sample_n):
        seq, labels, meta = task.get_tokens(metadata=True)
        rows.append(
            {
                "seq": seq,
                "answer": labels[-1],
                "path": meta["path"],
                "computed_answer": meta["answer"],
                "valid": meta["answer"] == labels[-1],
            }
        )
    return rows


def print_conditional_answer_table(rows, chars, feature_fn, feature_name):
    groups = defaultdict(Counter)
    for row in rows:
        groups[feature_fn(row)][row["answer"]] += 1
    print(f"\n{feature_name}")
    for key in sorted(groups):
        total = sum(groups[key].values())
        dist = {char: round(groups[key][char] / total, 3) for char in chars}
        print(f"  {key!r:>12} n={total:6d} {dist}")


def fit_majority_predictor(train_rows, feature_fn):
    global_majority = Counter(row["answer"] for row in train_rows).most_common(1)[0][0]
    table = defaultdict(Counter)
    for row in train_rows:
        table[feature_fn(row)][row["answer"]] += 1
    table = {key: counts.most_common(1)[0][0] for key, counts in table.items()}
    return lambda row: table.get(feature_fn(row), global_majority)


def heldout_acc(rows, feature_fn, seed):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(rows))
    split = len(rows) // 2
    train_rows = [rows[index] for index in perm[:split]]
    test_rows = [rows[index] for index in perm[split:]]
    predictor = fit_majority_predictor(train_rows, feature_fn)
    return sum(predictor(row) == row["answer"] for row in test_rows) / len(test_rows)


def acc_of_predictor(rows, pred_fn):
    return sum(pred_fn(row) == row["answer"] for row in rows) / len(rows)


def prefix_mode(seq, chars, k):
    counts = Counter(seq[:k])
    return max(chars, key=lambda char: (counts[char], -chars.index(char)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="phop_p32_seq256_a4_final")
    parser.add_argument("--split", default="train_constructive")
    parser.add_argument(
        "--data_root",
        default=os.environ.get("PHOP_DATA_ROOT", "/scratch/ab3u21/datasets/p-hop"),
    )
    parser.add_argument("--sample_n", type=int, default=50000)
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Read at most this many rows before optional subsampling.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--generate_if_missing",
        action="store_true",
        help="Generate synthetic constructive rows if the split file does not exist.",
    )
    args = parser.parse_args()

    spec = parse_task_name(args.task)
    chars = [chr(ord("a") + idx) for idx in range(spec["char_tokens"])]
    split_path = Path(args.data_root) / args.task / f"{args.split}.txt"

    print("=== p-hop shortcut statistics ===")
    print(f"task:       {args.task}")
    print(f"split:      {args.split}")
    print(f"hops:       {spec['hops']}")
    print(f"seq_len:    {spec['seq_len']}")
    print(f"alphabet:   {chars}")
    print(f"data path:  {split_path}")

    if split_path.exists():
        rows = load_rows_from_file(
            split_path,
            spec=spec,
            sample_n=args.sample_n,
            seed=args.seed,
            max_rows=args.max_rows,
        )
        print(f"source:     file ({len(rows)} sampled rows)")
    elif args.generate_if_missing:
        rows = generate_rows(spec, sample_n=args.sample_n, seed=args.seed)
        print(f"source:     generated ({len(rows)} rows)")
    else:
        raise FileNotFoundError(
            f"Missing {split_path}. Pass --generate_if_missing to sample from the generator."
        )

    valid_rate = sum(row["valid"] for row in rows) / len(rows)
    print(f"valid labels: {valid_rate:.6f}")
    if valid_rate != 1.0:
        print("WARNING: some labels do not match the computed p-hop answer.")

    answers = Counter(row["answer"] for row in rows)
    print("\nLabel distribution")
    print({char: round(answers[char] / len(rows), 4) for char in chars})
    print(f"majority baseline: {max(answers.values()) / len(rows):.4f}")

    print_conditional_answer_table(
        rows,
        chars,
        lambda row: row["seq"][-1],
        "P(answer | final query token)",
    )
    print_conditional_answer_table(
        rows,
        chars,
        lambda row: row["seq"][0],
        "P(answer | first token)",
    )

    print("\nSimple equality/exclusion checks")
    checks = [
        ("answer == final query", lambda row: row["answer"] == row["seq"][-1]),
        ("answer == previous token", lambda row: row["answer"] == row["seq"][-2]),
        ("answer == first token", lambda row: row["answer"] == row["seq"][0]),
        ("answer == hop31 token", lambda row: row["answer"] == row["path"][31][1]),
    ]
    for name, fn in checks:
        print(f"{name:28s} {sum(fn(row) for row in rows) / len(rows):.4f}")

    print("\nHeld-out shortcut baselines")
    features = [
        ("final query only", lambda row: row["seq"][-1]),
        ("last 2 tokens", lambda row: tuple(row["seq"][-2:])),
        ("last 4 tokens", lambda row: tuple(row["seq"][-4:])),
        ("first token", lambda row: row["seq"][0]),
        ("first 2 tokens", lambda row: tuple(row["seq"][:2])),
        ("first 3 tokens", lambda row: tuple(row["seq"][:3])),
        ("first 4 tokens", lambda row: tuple(row["seq"][:4])),
        ("first 8 tokens", lambda row: tuple(row["seq"][:8])),
        ("char counts", lambda row: tuple(Counter(row["seq"])[char] for char in chars)),
        (
            "query + char counts",
            lambda row: (row["seq"][-1], tuple(Counter(row["seq"])[char] for char in chars)),
        ),
        (
            "first8 + last4",
            lambda row: tuple(row["seq"][:8] + row["seq"][-4:]),
        ),
    ]
    for name, feature_fn in features:
        print(f"{name:24s} heldout_acc={heldout_acc(rows, feature_fn, args.seed):.4f}")

    print("\nStatic-position heuristics")
    static_position_acc = []
    for pos in range(spec["seq_len"]):
        static_position_acc.append(
            (acc_of_predictor(rows, lambda row, pos=pos: row["seq"][pos]), pos)
        )
    for acc, pos in sorted(static_position_acc, reverse=True)[:20]:
        print(f"pos={pos:3d} acc={acc:.4f}")

    print("\nPrefix mode heuristics")
    for k in [2, 4, 8, 16, 32, 64]:
        acc = acc_of_predictor(rows, lambda row, k=k: prefix_mode(row["seq"], chars, k))
        print(f"prefix_mode_{k:2d} acc={acc:.4f}")

    final_positions = np.array([row["path"][-1][0] for row in rows])
    print("\nFinal path index quantiles")
    for q in [0, 1, 5, 25, 50, 75, 95, 99, 100]:
        print(f"q{q:3d}: {np.percentile(final_positions, q):.1f}")
    for k in [2, 4, 8, 16, 32, 64]:
        print(f"P(final path index < {k:2d}) = {(final_positions < k).mean():.4f}")

    print("\nTruncated-hop answer agreement with true answer")
    for k in [0, 1, 2, 4, 8, 12, 16, 20, 24, 28, 30, 31, 32]:
        agree = 0
        valid = 0
        for row in rows:
            answer_k, _ = compute_path(row["seq"], k)
            if answer_k is not None:
                valid += 1
                agree += int(answer_k == row["answer"])
        print(f"k={k:2d}: agreement={agree / valid:.4f} valid={valid / len(rows):.4f}")


if __name__ == "__main__":
    main()
