#!/usr/bin/env python

import argparse
import ast
import glob
import json
import os
import re
from collections import defaultdict

ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
DICT_RE = re.compile(r"(\{.*\})")


def normalize_log_line(raw_line):
    line = ANSI_ESCAPE_RE.sub("", raw_line)
    return line.strip()


def parse_log(log_path):
    best_val = None
    test_metrics = None

    with open(log_path, "r") as f:
        for raw_line in f:
            line = normalize_log_line(raw_line)
            if line.startswith("on the test set:"):
                match = re.search(r"on the test set:\s*(\{.*\})", line)
                if match:
                    test_metrics = ast.literal_eval(match.group(1))
                continue

            if "'eval_accuracy'" not in line:
                continue

            match = DICT_RE.search(line)
            if not match:
                continue

            try:
                metrics = ast.literal_eval(match.group(1))
            except Exception:
                continue

            if best_val is None or metrics["eval_accuracy"] > best_val["eval_accuracy"]:
                best_val = metrics

    return best_val, test_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_root", required=True)
    parser.add_argument("--output_json", default="")
    args = parser.parse_args()

    rows = []
    grouped = defaultdict(list)
    for log_path in sorted(glob.glob(os.path.join(args.log_root, "*.log"))):
        best_val, test_metrics = parse_log(log_path)
        name = os.path.splitext(os.path.basename(log_path))[0]
        parts = name.split("__")
        if len(parts) < 3:
            continue
        variant = parts[0]
        pooling = ""
        lr = ""
        loop = "loop-ckpt"
        for part in parts:
            if part.startswith("loop-"):
                loop = part
            elif part.startswith("lr-"):
                lr = part.replace("lr-", "")
            elif part not in [variant, "diag"] and not pooling:
                pooling = part
        if variant == "loop1-stage-d":
            loop = "loop-ckpt"

        row = {
            "name": name,
            "variant": variant,
            "loop": loop,
            "pooling": pooling,
            "lr": lr,
            "best_val": best_val,
            "test": test_metrics,
            "log_path": log_path,
        }
        rows.append(row)
        grouped[(variant, loop, pooling)].append(row)

    selected = []
    for key, candidates in grouped.items():
        candidates = [c for c in candidates if c["best_val"] is not None and c["test"] is not None]
        if not candidates:
            continue
        best = max(candidates, key=lambda c: c["best_val"]["eval_accuracy"])
        selected.append(best)

    selected.sort(key=lambda row: (row["variant"], row["loop"], row["pooling"]))

    for row in selected:
        print(
            "\t".join(
                [
                    row["variant"],
                    row["loop"],
                    row["pooling"],
                    row["lr"],
                    f"val_acc={row['best_val']['eval_accuracy']:.6f}",
                    f"test_acc={row['test']['eval_accuracy']:.6f}",
                    f"test_f1={row['test']['eval_f1']:.6f}",
                ]
            )
        )

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(selected, f, indent=2)


if __name__ == "__main__":
    main()
