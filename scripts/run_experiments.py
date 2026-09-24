#!/usr/bin/env python3
"""One-command runner for all bundled in-domain and zero-shot experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import defaults as D
from src.train_eval import run_all_experiments


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run all bundled DoLLM in-domain and zero-shot evaluations."
    )
    parser.add_argument("--syn-dir", default=str(PROJECT_ROOT / "toy_datasets" / "Syn"))
    parser.add_argument("--udp-dir", default=str(PROJECT_ROOT / "toy_datasets" / "UDP"))
    parser.add_argument("--model-name-or-path", default=D.DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", default="outputs/experiments")
    parser.add_argument("--seeds", nargs="+", type=int, default=D.SEEDS)
    parser.add_argument("--epochs", type=int, default=D.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=D.BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=D.NUM_WORKERS)
    parser.add_argument("--binning-num", type=int, default=D.BINNING_NUM)
    parser.add_argument("--num-training-samples", type=int, default=D.NUM_TRAINING_SAMPLES)
    parser.add_argument("--learning-rate", type=float, default=D.LEARNING_RATE)
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_dir)
    common = dict(
        model_name_or_path=args.model_name_or_path,
        num_flows=args.binning_num,
        num_training_samples=args.num_training_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    print("\n=== Training once on SYN, then evaluating SYN and UDP ===")
    print("=== Training once on UDP, then evaluating UDP and SYN ===")
    all_rows = run_all_experiments(
        syn_dataset_dir=args.syn_dir,
        udp_dataset_dir=args.udp_dir,
        seeds=args.seeds,
        **common,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    result_frame = pd.DataFrame(all_rows)
    result_frame.to_csv(output_root / "all_results.csv", index=False)
    summary = result_frame.groupby(["setting", "source_dataset", "target_dataset"])[
        ["f1", "accuracy", "precision", "recall", "auc"]
    ].agg(["mean", "std"])
    summary.to_csv(output_root / "all_summary.csv")
    print(f"\nAll experiments finished. Runs={len(all_rows)}")
    print(f"Combined results: {output_root / 'all_results.csv'}")
    print(f"Combined summary: {output_root / 'all_summary.csv'}")


if __name__ == "__main__":
    main()
