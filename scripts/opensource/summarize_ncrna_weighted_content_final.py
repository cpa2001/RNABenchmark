#!/usr/bin/env python

import argparse
import json
import os
from statistics import mean
from typing import Dict, List, Optional


SEEDS = [666, 42, 3407]
SETTINGS = [
    ("loop1-stage-d", "loops-ckpt", "loop1-stage-d-weighted-content-final"),
    ("stage-d-100k__loop-1", "loops-1", "stage-d-100k-weighted-content-final"),
    ("stage-d-100k__loop-2", "loops-2", "stage-d-100k-weighted-content-final"),
    ("stage-d-100k__loop-3", "loops-3", "stage-d-100k-weighted-content-final"),
]


def load_json(path: str) -> Dict[str, object]:
    with open(path, "r") as f:
        return json.load(f)


def build_result_path(results_root: str, variant: str, strategy: str, loop_label: str, lr: str, seed: int) -> str:
    return os.path.join(
        results_root,
        variant,
        strategy,
        loop_label,
        "frozen-head",
        f"lr-{lr}",
        str(seed),
        "results",
        f"ecorna_ncrna_frozen_head_lr-{lr}",
        "test_results.json",
    )


def mean_matrix(mats: List[List[List[float]]]) -> List[List[float]]:
    rows = len(mats[0])
    cols = len(mats[0][0])
    out = []
    for r in range(rows):
        row = []
        for c in range(cols):
            row.append(mean(mat[r][c] for mat in mats))
        out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default="./outputs/ft/rna-all/NoncodingRNAFamily/ecorna")
    parser.add_argument("--strategy", default="weighted_layer_content")
    parser.add_argument("--lr", default="1e-3")
    parser.add_argument("--output_json", default="")
    args = parser.parse_args()

    results_root = os.path.abspath(args.results_root)
    rows = []
    for setting_name, loop_label, variant in SETTINGS:
        seed_rows = []
        for seed in SEEDS:
            path = build_result_path(results_root, variant, args.strategy, loop_label, args.lr, seed)
            if not os.path.exists(path):
                continue
            metrics = load_json(path)
            seed_rows.append(
                {
                    "seed": seed,
                    "path": path,
                    "accuracy": metrics["eval_accuracy"],
                    "f1": metrics["eval_f1"],
                    "pooler_mean_max_weight": metrics.get("pooler_mean_max_weight"),
                    "pooler_effective_cells": metrics.get("pooler_effective_cells"),
                    "pooler_max_abs_raw_logit": metrics.get("pooler_max_abs_raw_logit"),
                    "pooler_mean_weights": metrics.get("pooler_mean_weights"),
                }
            )

        summary: Dict[str, object] = {
            "setting": setting_name,
            "variant": variant,
            "strategy": args.strategy,
            "lr": args.lr,
            "count": len(seed_rows),
            "seed_rows": seed_rows,
        }
        if seed_rows:
            summary.update(
                {
                    "mean_accuracy": mean(row["accuracy"] for row in seed_rows),
                    "mean_f1": mean(row["f1"] for row in seed_rows),
                    "mean_pooler_max_weight": mean(row["pooler_mean_max_weight"] for row in seed_rows),
                    "mean_pooler_effective_cells": mean(row["pooler_effective_cells"] for row in seed_rows),
                    "mean_pooler_max_abs_raw_logit": mean(row["pooler_max_abs_raw_logit"] for row in seed_rows),
                    "mean_pooler_weights": mean_matrix([row["pooler_mean_weights"] for row in seed_rows]),
                }
            )
        rows.append(summary)

    payload = {
        "results_root": results_root,
        "strategy": args.strategy,
        "lr": args.lr,
        "rows": rows,
    }

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2)

    print("setting\tcount\tmean_acc\tmean_f1\tmean_max_weight\tmean_effective_cells\tmean_max_raw_logit")
    for row in rows:
        if row["count"] == 0:
            print(f"{row['setting']}\t0\tNA\tNA\tNA\tNA\tNA")
            continue
        print(
            "\t".join(
                [
                    row["setting"],
                    str(row["count"]),
                    f"{row['mean_accuracy']:.6f}",
                    f"{row['mean_f1']:.6f}",
                    f"{row['mean_pooler_max_weight']:.6f}",
                    f"{row['mean_pooler_effective_cells']:.6f}",
                    f"{row['mean_pooler_max_abs_raw_logit']:.6f}",
                ]
            )
        )


if __name__ == "__main__":
    main()
