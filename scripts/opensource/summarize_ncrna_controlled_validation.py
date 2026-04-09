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
        seed = ""
        loop = ""
        pooling = ""
        lr = ""
        for part in parts:
            if part.startswith("seed-"):
                seed = part.replace("seed-", "")
            elif part.startswith("loop-"):
                loop = part
            elif part.startswith("lr-"):
                lr = part.replace("lr-", "")
            elif not part.startswith("seed-") and not part.startswith("loop-") and not part.startswith("lr-") and not pooling:
                pooling = part

        row = {
            "name": name,
            "seed": seed,
            "loop": loop,
            "pooling": pooling,
            "lr": lr,
            "best_val": best_val,
            "test": test_metrics,
            "log_path": log_path,
        }
        rows.append(row)
        grouped[(loop, pooling)].append(row)

    summary = {}
    for key, candidates in grouped.items():
        valid_candidates = [c for c in candidates if c["best_val"] is not None and c["test"] is not None]
        if not valid_candidates:
            continue
        valid_candidates.sort(key=lambda row: row["seed"])
        test_acc = [row["test"]["eval_accuracy"] for row in valid_candidates]
        summary[f"{key[0]}__{key[1]}"] = {
            "count": len(valid_candidates),
            "mean_test_acc": sum(test_acc) / len(test_acc),
            "min_test_acc": min(test_acc),
            "max_test_acc": max(test_acc),
            "rows": valid_candidates,
        }

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump({"rows": rows, "summary": summary}, f, indent=2)

    for key in sorted(summary):
        item = summary[key]
        print(
            "\t".join(
                [
                    key,
                    f"count={item['count']}",
                    f"mean_test_acc={item['mean_test_acc']:.6f}",
                    f"min_test_acc={item['min_test_acc']:.6f}",
                    f"max_test_acc={item['max_test_acc']:.6f}",
                ]
            )
        )


if __name__ == "__main__":
    main()
