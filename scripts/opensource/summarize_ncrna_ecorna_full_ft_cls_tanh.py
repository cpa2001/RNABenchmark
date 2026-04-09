#!/usr/bin/env python

import argparse
import json
import os
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional


SEEDS = [666, 42, 3407]
LOOPS = ["loops-1", "loops-2", "loops-3"]
LEGACY_CONTROL = ("loops--1", [666])


def load_json(path: str) -> Dict[str, object]:
    with open(path, "r") as f:
        return json.load(f)


def load_legacy_json(path: str) -> Dict[str, object]:
    text = Path(path).read_text()
    decoder = json.JSONDecoder()
    idx = 0
    last: Optional[Dict[str, object]] = None
    while idx < len(text):
        while idx < len(text) and text[idx] != "{":
            idx += 1
        if idx >= len(text):
            break
        try:
            obj, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            idx += 1
            continue
        if isinstance(obj, dict):
            last = obj
        idx = end
    if last is None:
        raise ValueError(f"Could not parse a JSON object from {path}")
    return last


def build_result_path(results_root: str, loop_label: str, seed: int) -> str:
    return os.path.join(
        results_root,
        "cls_tanh",
        loop_label,
        str(seed),
        "results",
        "ecorna_ncrna",
        "test_results.json",
    )


def collect_rows(results_root: str, loop_label: str, seeds: List[int]) -> List[Dict[str, object]]:
    rows = []
    for seed in seeds:
        path = build_result_path(results_root, loop_label, seed)
        if not os.path.exists(path):
            continue
        metrics = load_legacy_json(path)
        rows.append(
            {
                "seed": seed,
                "path": path,
                "accuracy": float(metrics["eval_accuracy"]),
                "f1": float(metrics["eval_f1"]),
                "epoch": float(metrics.get("epoch", 0.0)),
            }
        )
    return rows


def summarize_group(loop_label: str, rows: List[Dict[str, object]]) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "loop_label": loop_label,
        "count": len(rows),
        "seed_rows": rows,
    }
    if rows:
        payload.update(
            {
                "mean_accuracy": mean(row["accuracy"] for row in rows),
                "mean_f1": mean(row["f1"] for row in rows),
                "mean_epoch": mean(row["epoch"] for row in rows),
            }
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_root",
        default="./outputs/ft/rna-all/NoncodingRNAFamily/ecorna",
    )
    parser.add_argument("--output_json", default="")
    args = parser.parse_args()

    results_root = os.path.abspath(args.results_root)
    payload: Dict[str, object] = {
        "results_root": results_root,
        "strategy": "cls_tanh",
        "mode": "full-ft",
        "rows": [],
    }

    for loop_label in LOOPS:
        rows = collect_rows(results_root, loop_label, SEEDS)
        payload["rows"].append(summarize_group(loop_label, rows))

    control_label, control_seeds = LEGACY_CONTROL
    payload["legacy_control"] = summarize_group(
        control_label,
        collect_rows(results_root, control_label, control_seeds),
    )

    valid_rows = [row for row in payload["rows"] if row["count"] == len(SEEDS)]
    if valid_rows:
        best = max(valid_rows, key=lambda row: (row["mean_f1"], row["mean_accuracy"]))
        payload["best_loop_by_f1"] = {
            "loop_label": best["loop_label"],
            "mean_accuracy": best["mean_accuracy"],
            "mean_f1": best["mean_f1"],
        }

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2)

    print("loop\tcount\tmean_acc\tmean_f1\tmean_epoch")
    for row in payload["rows"]:
        if row["count"] == 0:
            print(f"{row['loop_label']}\t0\tNA\tNA\tNA")
        else:
            print(
                "\t".join(
                    [
                        row["loop_label"],
                        str(row["count"]),
                        f"{row['mean_accuracy']:.6f}",
                        f"{row['mean_f1']:.6f}",
                        f"{row['mean_epoch']:.6f}",
                    ]
                )
            )
    control = payload["legacy_control"]
    if control["count"]:
        print(
            "legacy_control\t{count}\t{acc:.6f}\t{f1:.6f}\t{epoch:.6f}".format(
                count=control["count"],
                acc=control["mean_accuracy"],
                f1=control["mean_f1"],
                epoch=control["mean_epoch"],
            )
        )


if __name__ == "__main__":
    main()
