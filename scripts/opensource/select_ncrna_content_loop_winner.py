#!/usr/bin/env python

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict


ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
DICT_RE = re.compile(r"(\{.*\})")
EXPECTED_SETTINGS = [
    ("loop1-stage-d", "loop-ckpt"),
    ("stage-d-100k", "loop-1"),
    ("stage-d-100k", "loop-2"),
    ("stage-d-100k", "loop-3"),
]
ALLOWED_STRATEGIES = {"content_mean", "loop_mean_content"}


def normalize_log_line(raw_line):
    return ANSI_ESCAPE_RE.sub("", raw_line).strip()


def safe_parse_metrics_dict(raw_dict):
    return eval(  # noqa: S307
        raw_dict,
        {"__builtins__": {}},
        {"nan": float("nan"), "inf": float("inf")},
    )


def best_val_sort_key(metrics):
    return (
        metrics.get("eval_f1", float("-inf")),
        metrics.get("eval_accuracy", float("-inf")),
    )


def parse_log(log_path):
    best_val = None
    test_metrics = None

    with open(log_path, "r") as f:
        for raw_line in f:
            line = normalize_log_line(raw_line)
            if line.startswith("on the test set:"):
                match = re.search(r"on the test set:\s*(\{.*\})", line)
                if match:
                    try:
                        test_metrics = safe_parse_metrics_dict(match.group(1))
                    except Exception:
                        test_metrics = None
                continue

            if "'eval_accuracy'" not in line:
                continue

            match = DICT_RE.search(line)
            if not match:
                continue

            try:
                metrics = safe_parse_metrics_dict(match.group(1))
            except Exception:
                continue

            if best_val is None or best_val_sort_key(metrics) > best_val_sort_key(best_val):
                best_val = metrics

    return best_val, test_metrics


def parse_job_name(log_path):
    name = os.path.splitext(os.path.basename(log_path))[0]
    parts = name.split("__")
    variant = parts[0]
    loop = "loop-ckpt"
    strategy = ""
    lr = ""

    for part in parts[1:]:
        if part.startswith("loop-"):
            loop = part
        elif part.startswith("lr-"):
            lr = part.replace("lr-", "", 1)
        elif not strategy:
            strategy = part

    return {
        "name": name,
        "variant": variant,
        "loop": loop,
        "strategy": strategy,
        "lr": lr,
    }


def candidate_sort_key(item):
    return (
        item["mean_val_f1"],
        item["mean_val_accuracy"],
        -item["lr_numeric"],
        1 if item["strategy"] == "content_mean" else 0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_root", required=True)
    parser.add_argument("--output_json", default="")
    args = parser.parse_args()

    rows = []
    ignored_rows = []
    grouped = defaultdict(dict)

    for log_path in sorted(glob.glob(os.path.join(args.log_root, "*.log"))):
        job = parse_job_name(log_path)
        best_val, test_metrics = parse_log(log_path)
        row = {
            **job,
            "best_val": best_val,
            "test": test_metrics,
            "log_path": log_path,
        }
        if job["strategy"] not in ALLOWED_STRATEGIES:
            ignored_rows.append(row)
            continue
        rows.append(row)
        grouped[(job["strategy"], job["lr"])][(job["variant"], job["loop"])] = row

    candidates = []
    for (strategy, lr), by_setting in grouped.items():
        valid_rows = []
        missing_settings = []
        for setting in EXPECTED_SETTINGS:
            row = by_setting.get(setting)
            if row is None or row["best_val"] is None or row["test"] is None:
                missing_settings.append(f"{setting[0]}__{setting[1]}")
            else:
                valid_rows.append(row)

        mean_val_accuracy = None
        mean_val_f1 = None
        if valid_rows:
            mean_val_accuracy = sum(r["best_val"]["eval_accuracy"] for r in valid_rows) / len(valid_rows)
            mean_val_f1 = sum(r["best_val"]["eval_f1"] for r in valid_rows) / len(valid_rows)

        candidate = {
            "strategy": strategy,
            "lr": lr,
            "lr_numeric": float(lr),
            "num_settings": len(valid_rows),
            "complete": len(valid_rows) == len(EXPECTED_SETTINGS),
            "mean_val_accuracy": mean_val_accuracy,
            "mean_val_f1": mean_val_f1,
            "missing_settings": missing_settings,
            "rows": sorted(valid_rows, key=lambda row: (row["variant"], row["loop"])),
        }
        candidates.append(candidate)

    complete_candidates = [c for c in candidates if c["complete"]]
    winner = max(complete_candidates, key=candidate_sort_key) if complete_candidates else None

    candidates.sort(
        key=lambda item: (
            item["complete"],
            item["mean_val_f1"] if item["mean_val_f1"] is not None else -1.0,
            item["mean_val_accuracy"] if item["mean_val_accuracy"] is not None else -1.0,
            -item["lr_numeric"],
            1 if item["strategy"] == "content_mean" else 0,
        ),
        reverse=True,
    )

    payload = {
        "log_root": args.log_root,
        "expected_settings": [f"{variant}__{loop}" for variant, loop in EXPECTED_SETTINGS],
        "rows": rows,
        "ignored_rows": ignored_rows,
        "candidates": candidates,
        "winner": winner,
    }

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2)

    print("strategy\tlr\tsettings\tmean_val_f1\tmean_val_acc\tstatus")
    for item in candidates:
        status = "complete" if item["complete"] else f"missing={','.join(item['missing_settings'])}"
        mean_val_acc = "NA" if item["mean_val_accuracy"] is None else f"{item['mean_val_accuracy']:.6f}"
        mean_val_f1 = "NA" if item["mean_val_f1"] is None else f"{item['mean_val_f1']:.6f}"
        print(
            "\t".join(
                [
                    item["strategy"],
                    item["lr"],
                    str(item["num_settings"]),
                    mean_val_f1,
                    mean_val_acc,
                    status,
                ]
            )
        )

    if winner is None:
        print("No complete pilot candidate found. Finish all 16 pilot jobs before selecting a winner.", file=sys.stderr)
        sys.exit(1)

    print()
    print(
        "winner\t{strategy}\tlr={lr}\tmean_val_f1={mean_val_f1:.6f}\tmean_val_acc={mean_val_accuracy:.6f}".format(
            **winner
        )
    )


if __name__ == "__main__":
    main()
