import argparse
import json
import math
import os
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

import config
import distributed
import models

from generate import IGNORE_INDEX, InductionHopsFinalAnswerTask
from phop_data import (
    DEFAULT_PHOP_TASK,
    default_data_root,
    evaluate_phop_model,
    get_phop_task_spec,
    make_phop_dataloader,
)


POSITION_BINS = [
    ("final_path_index < 4", 0, 4),
    ("4 <= final_path_index < 8", 4, 8), 
    ("8 <= final_path_index < 16", 8, 16),
    ("final_path_index >= 16", 16, None),
]


def none_or_str(value):
    if value == "None":
        return None
    return value


def latest_checkpoint_file(checkpoint_dir):
    checkpoints = [
        file for file in os.listdir(checkpoint_dir)
        if file.startswith("ckpt_") and file.endswith(".pt")
    ]
    if not checkpoints:
        return None
    return max(
        checkpoints,
        key=lambda file: int(file.split("ckpt_")[1].split(".pt")[0]),
    )


def best_checkpoint_file(checkpoint_dir):
    preferred = "best_val_constructive_acc.pt"
    preferred_path = os.path.join(checkpoint_dir, preferred)
    if os.path.isfile(preferred_path):
        return preferred

    best_metrics_path = os.path.join(checkpoint_dir, "best_metrics.json")
    if os.path.isfile(best_metrics_path):
        with open(best_metrics_path, encoding="utf-8") as handle:
            best_metrics = json.load(handle)
        checkpoint = best_metrics.get("checkpoint")
        if checkpoint is not None and os.path.isfile(os.path.join(checkpoint_dir, checkpoint)):
            return checkpoint

    return None


def apply_phop_task_shape(args, spec):
    if getattr(args, "sequence_length", None) is None:
        args.sequence_length = spec.minimum_sequence_length
    elif int(args.sequence_length) < spec.minimum_sequence_length:
        args.sequence_length = spec.minimum_sequence_length

    args.vocab_size = spec.vocab_size
    args.dataset = spec.name


def load_summary_args(args):
    if args.checkpoint is None or os.path.isfile(args.checkpoint):
        return args

    summary_path = os.path.join(args.checkpoint, "summary.json")
    if not os.path.isfile(summary_path):
        return args

    cli_overrides = {
        "checkpoint": args.checkpoint,
        "checkpoint_filename": args.checkpoint_filename,
        "config_format": args.config_format,
        "phop_eval_splits": args.phop_eval_splits,
        "phop_eval_batch_size": args.phop_eval_batch_size,
        "phop_eval_max_batches": args.phop_eval_max_batches,
        "phop_data_root": args.phop_data_root,
        "phop_num_workers": args.phop_num_workers,
        "diagnostics": args.diagnostics,
        "diagnostic_max_examples": args.diagnostic_max_examples,
        "output_json": args.output_json,
        "output_jsonl": args.output_jsonl,
    }

    with open(summary_path, encoding="utf-8") as handle:
        summary = json.load(handle)

    for key, value in summary.get("args", {}).items():
        if key not in {"device", "dtype"}:
            setattr(args, key, value)

    for key, value in cli_overrides.items():
        if value is not None:
            setattr(args, key, value)

    return args


def get_args(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--checkpoint", type=none_or_str, default=None)
    parser.add_argument("--checkpoint_filename", default=None)
    parser.add_argument("--config_format", default="base")

    parser.add_argument("--phop_task", default=DEFAULT_PHOP_TASK)
    parser.add_argument("--phop_data_root", default=str(default_data_root()))
    parser.add_argument("--phop_eval_splits", nargs="+", default=["val_constructive", "test_constructive"])
    parser.add_argument("--phop_eval_batch_size", type=int, default=None)
    parser.add_argument("--phop_eval_max_batches", type=int, default=None)
    parser.add_argument("--phop_num_workers", type=int, default=0)

    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--diagnostic_max_examples", type=int, default=None)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_jsonl", default=None)


    args, rem_args = parser.parse_known_args(argv)

    if args.checkpoint is not None and os.path.isfile(args.checkpoint):
        checkpoint_dir, checkpoint_file = os.path.split(args.checkpoint)
        args.checkpoint = checkpoint_dir
        if args.checkpoint_filename is None:
            args.checkpoint_filename = checkpoint_file

    args = load_summary_args(args)

    if args.checkpoint is not None and args.checkpoint_filename is None:
        args.checkpoint_filename = best_checkpoint_file(args.checkpoint)

    if args.checkpoint is not None and args.checkpoint_filename is None:
        args.checkpoint_filename = latest_checkpoint_file(args.checkpoint)

    args.distributed_backend = None
    return config.parse_args_with_format(
        format=args.config_format,
        base_parser=argparse.ArgumentParser(allow_abbrev=False),
        args=rem_args,
        namespace=args,
    )


def load_model(args):
    model = models.make_model_from_args(args).to(args.device)

    if args.checkpoint is None:
        model.eval()
        return model

    if args.checkpoint_filename is None:
        raise FileNotFoundError(f"No checkpoint file found in {args.checkpoint}")

    checkpoint_path = os.path.join(args.checkpoint, args.checkpoint_filename)
    checkpoint = torch.load(checkpoint_path, map_location=args.device)

    state = {
        key.replace("_orig_mod.", ""): value
        for key, value in checkpoint["model"].items()
    }
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def decode_ids(ids, vocab):
    return [vocab[int(idx)] for idx in ids]


def bucket_name_for_final_path_index(final_path_index):
    for name, lower, upper in POSITION_BINS:
        if final_path_index >= lower and (upper is None or final_path_index < upper):
            return name
    raise RuntimeError(f"No bucket for final_path_index={final_path_index}")


def new_bucket_stats(vocab):
    return {
        "count": 0,
        "correct": 0,
        "failures": 0,
        "answer_distribution": Counter(),
        "prediction_distribution": Counter(),
        "confusion": {
            true_token: Counter()
            for true_token in vocab
            if not true_token.startswith("<") and true_token != "<pad>"
        },
    }


def counter_to_distribution(counter):
    total = sum(counter.values())
    if total == 0:
        return {}
    return {
        key: {
            "count": int(value),
            "pct": float(value / total),
        }
        for key, value in sorted(counter.items())
    }


def finalize_bucket_stats(bucket_stats, total_examples, total_failures):
    result = {}

    for name, stats in bucket_stats.items():
        count = stats["count"]
        correct = stats["correct"]
        failures = stats["failures"]

        confusion = {}
        for true_token, pred_counter in stats["confusion"].items():
            if sum(pred_counter.values()) > 0:
                confusion[true_token] = counter_to_distribution(pred_counter)

        result[name] = {
            "count": int(count),
            "dataset_pct": float(count / total_examples) if total_examples else math.nan,
            "acc": float(correct / count) if count else math.nan,
            "failure_count": int(failures),
            "failure_pct_of_bucket": float(failures / count) if count else math.nan,
            "failure_pct_of_all_failures": (
                float(failures / total_failures) if total_failures else math.nan
            ),
            "answer_distribution": counter_to_distribution(stats["answer_distribution"]),
            "prediction_distribution": counter_to_distribution(stats["prediction_distribution"]),
            "confusion": confusion,
        }

    return result


def make_path_generator(spec):
    return InductionHopsFinalAnswerTask(
        seq_len=spec.seq_len,
        char_tokens=spec.char_tokens,
        min_hops=spec.min_hops,
        max_hops=spec.max_hops,
        include_hop_token=spec.include_hop_token,
        avoid_adjacent_repeats=spec.avoid_adjacent_repeats,
        sampling_strategy=spec.sampling_strategy,
    )


@torch.no_grad()
def run_standard_eval(args, model, spec, type_ctx, device_type):
    data_root = Path(args.phop_data_root)
    eval_batch_size = args.phop_eval_batch_size or args.batch_size

    stats = {}
    for split in args.phop_eval_splits:
        dataloader = make_phop_dataloader(
            data_root=data_root,
            task=spec.name,
            split=split,
            spec=spec,
            sequence_length=args.sequence_length,
            batch_size=eval_batch_size,
            shuffle=False,
            seed=args.seed,
            num_workers=args.phop_num_workers,
            pin_memory=device_type == "cuda",
        )
        stats[split] = evaluate_phop_model(
            model,
            dataloader,
            device=args.device,
            max_batches=args.phop_eval_max_batches,
            ctx=type_ctx,
        )

    return stats


@torch.no_grad()
def run_diagnostic_eval_split(args, model, spec, split, type_ctx, device_type):
    data_root = Path(args.phop_data_root)
    eval_batch_size = args.phop_eval_batch_size or args.batch_size
    vocab = spec.vocab

    dataloader = make_phop_dataloader(
        data_root=data_root,
        task=spec.name,
        split=split,
        spec=spec,
        sequence_length=args.sequence_length,
        batch_size=eval_batch_size,
        shuffle=False,
        seed=args.seed,
        num_workers=args.phop_num_workers,
        pin_memory=device_type == "cuda",
    )

    path_generator = make_path_generator(spec)

    bucket_stats = {
        name: new_bucket_stats(vocab)
        for name, _, _ in POSITION_BINS
    }

    rows = []
    final_path_positions = []
    total = 0
    correct = 0

    for batch_idx, batch in enumerate(dataloader):
        if args.phop_eval_max_batches is not None and batch_idx >= args.phop_eval_max_batches:
            break
        if args.diagnostic_max_examples is not None and total >= args.diagnostic_max_examples:
            break

        inputs = batch["input_id"].to(args.device)
        labels = batch["label"].to(args.device)

        with type_ctx:
            outputs = model(inputs, targets=labels, get_logits=True)

        logits = outputs["logits"]
        final_logits = logits[:, -1, :]
        probs = torch.softmax(final_logits, dim=-1)
        pred_ids = probs.argmax(dim=-1)
        pred_confs = probs.max(dim=-1).values

        inputs_cpu = batch["input_id"].cpu()
        labels_cpu = batch["label"].cpu()
        pred_ids_cpu = pred_ids.detach().cpu()
        pred_confs_cpu = pred_confs.detach().cpu()

        for row_idx in range(inputs_cpu.shape[0]):
            if args.diagnostic_max_examples is not None and total >= args.diagnostic_max_examples:
                break

            input_tokens = decode_ids(inputs_cpu[row_idx].tolist(), vocab)
            raw_seq = (
                input_tokens[1:1 + spec.seq_len]
                if spec.include_hop_token
                else input_tokens[:spec.seq_len]
            )

            computed_answer, path = path_generator._compute_final_answer(raw_seq, spec.max_hops)
            final_path_index = int(path[-1][0])
            final_path_positions.append(final_path_index)

            label_positions = (labels_cpu[row_idx] != IGNORE_INDEX).nonzero(as_tuple=False)
            if len(label_positions) != 1:
                raise ValueError(
                    f"Expected one supervised p-hop label, got {len(label_positions)}"
                )

            true_pos = int(label_positions[0].item())
            true_id = int(labels_cpu[row_idx, true_pos].item())
            true_token = vocab[true_id]

            pred_id = int(pred_ids_cpu[row_idx].item())
            pred_token = vocab[pred_id]
            pred_conf = float(pred_confs_cpu[row_idx].item())

            is_correct = pred_token == true_token
            total += 1
            correct += int(is_correct)

            bucket_name = bucket_name_for_final_path_index(final_path_index)
            stats = bucket_stats[bucket_name]
            stats["count"] += 1
            stats["correct"] += int(is_correct)
            stats["failures"] += int(not is_correct)
            stats["answer_distribution"][true_token] += 1
            stats["prediction_distribution"][pred_token] += 1
            stats["confusion"][true_token][pred_token] += 1

            rows.append({
                "split": split,
                "example_idx": total - 1,
                "correct": bool(is_correct),
                "true_answer": true_token,
                "computed_answer": computed_answer,
                "pred_answer": pred_token,
                "pred_confidence": pred_conf,
                "hops": int(spec.max_hops),
                "final_path_index": final_path_index,
                "final_path_fraction": float(final_path_index / spec.seq_len),
                "bucket": bucket_name,
                "path": [
                    {
                        "hop": int(hop_idx),
                        "position": int(position),
                        "token": token,
                    }
                    for hop_idx, (position, token) in enumerate(path)
                ],
                "input": "".join(raw_seq),
            })

    total_failures = total - correct

    summary = {
        "num_examples": int(total),
        "acc": float(correct / total) if total else math.nan,
        "num_failures": int(total_failures),
        "final_path_index": {
            "mean": float(np.mean(final_path_positions)) if final_path_positions else math.nan,
            "median": float(np.median(final_path_positions)) if final_path_positions else math.nan,
            "min": int(np.min(final_path_positions)) if final_path_positions else None,
            "max": int(np.max(final_path_positions)) if final_path_positions else None,
        },
        "position_buckets": finalize_bucket_stats(
            bucket_stats,
            total_examples=total,
            total_failures=total_failures,
        ),
    }

    return summary, rows


def run_diagnostics(args, model, spec, type_ctx, device_type):
    summaries = {}
    rows = []

    for split in args.phop_eval_splits:
        split_summary, split_rows = run_diagnostic_eval_split(
            args=args,
            model=model,
            spec=spec,
            split=split,
            type_ctx=type_ctx,
            device_type=device_type,
        )
        summaries[split] = split_summary
        rows.extend(split_rows)

    return summaries, rows


def main(args):
    args.distributed_backend = "None"

    args.device = torch.device(args.device)


    args.device = torch.device(args.device)
    device_type = "cuda" if "cuda" in str(args.device) else "cpu"
    if device_type == "cuda":
        torch.cuda.set_device(args.device)

    type_ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(
        device_type=device_type,
        dtype=args.dtype,
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    spec = get_phop_task_spec(args.phop_task)
    apply_phop_task_shape(args, spec)

    model = load_model(args)

    result = {
        "checkpoint": args.checkpoint,
        "checkpoint_filename": args.checkpoint_filename,
        "task": spec.name,
        "phop_data_root": str(args.phop_data_root),
        "eval_splits": list(args.phop_eval_splits),
        "standard_eval": run_standard_eval(
            args=args,
            model=model,
            spec=spec,
            type_ctx=type_ctx,
            device_type=device_type,
        ),
    }

    diagnostic_rows = []
    if args.diagnostics:
        diagnostic_summary, diagnostic_rows = run_diagnostics(
            args=args,
            model=model,
            spec=spec,
            type_ctx=type_ctx,
            device_type=device_type,
        )
        result["diagnostics"] = diagnostic_summary

    # if distributed_backend.is_master_process():
    #     print(json.dumps(result, indent=2))

    #     if args.output_json is not None:
    #         output_path = Path(args.output_json)
    #         output_path.parent.mkdir(parents=True, exist_ok=True)
    #         with output_path.open("w", encoding="utf-8") as handle:
    #             json.dump(result, handle, indent=2)

    #     if args.output_jsonl is not None:
    #         output_path = Path(args.output_jsonl)
    #         output_path.parent.mkdir(parents=True, exist_ok=True)
    #         with output_path.open("w", encoding="utf-8") as handle:
    #             for row in diagnostic_rows:
    #                 handle.write(json.dumps(row) + "\n")

    # distributed_backend.finalize()

    print(json.dumps(result, indent=2))

    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)

    if args.output_jsonl is not None:
        output_path = Path(args.output_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for row in diagnostic_rows:
                handle.write(json.dumps(row) + "\n")



if __name__ == "__main__":
    main(get_args())

