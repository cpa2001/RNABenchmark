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
FINAL_SEEDS = [666, 42, 3407]
SETTINGS = [
    {"label": "loop1-stage-d", "loop": "loop-ckpt", "loop_output": "loops-ckpt"},
    {"label": "stage-d-100k", "loop": "loop-1", "loop_output": "loops-1"},
    {"label": "stage-d-100k", "loop": "loop-2", "loop_output": "loops-2"},
    {"label": "stage-d-100k", "loop": "loop-3", "loop_output": "loops-3"},
]


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


def parse_name(log_path):
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

    return variant, loop, strategy, lr


def select_from_logs(log_root, variant, loop, strategy):
    candidates = []
    for log_path in glob.glob(os.path.join(log_root, "*.log")):
        job_variant, job_loop, job_strategy, job_lr = parse_name(log_path)
        if (job_variant, job_loop, job_strategy) != (variant, loop, strategy):
            continue
        best_val, test_metrics = parse_log(log_path)
        if best_val is None or test_metrics is None:
            continue
        candidates.append(
            {
                "lr": job_lr,
                "best_val": best_val,
            }
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: (
            item["best_val"]["eval_f1"],
            item["best_val"]["eval_accuracy"],
            -float(item["lr"]),
        ),
        reverse=True,
    )
    return candidates[0]


def load_metrics(path):
    with open(path, "r") as f:
        return json.load(f)


def aggregate_result_paths(paths):
    rows = []
    fallback_batches = 0
    fallback_examples = 0
    for seed, path in paths:
        if not os.path.exists(path):
            continue
        metrics = load_metrics(path)
        fallback_batches += int(metrics.get("content_mean_fallback_batches", 0))
        fallback_examples += int(metrics.get("content_mean_fallback_examples", 0))
        rows.append({"seed": seed, "path": path, "metrics": metrics})

    if not rows:
        return None

    mean_accuracy = sum(row["metrics"]["eval_accuracy"] for row in rows) / len(rows)
    mean_f1 = sum(row["metrics"]["eval_f1"] for row in rows) / len(rows)
    return {
        "count": len(rows),
        "mean_test_accuracy": mean_accuracy,
        "mean_test_f1": mean_f1,
        "content_mean_fallback_batches": fallback_batches,
        "content_mean_fallback_examples": fallback_examples,
        "rows": rows,
    }


def build_result_path(root, variant, strategy, loop_output, lr, seed):
    return os.path.join(
        root,
        variant,
        strategy,
        loop_output,
        "frozen-head",
        f"lr-{lr}",
        str(seed),
        "results",
        f"ecorna_ncrna_frozen_head_lr-{lr}",
        "test_results.json",
    )


def maybe_latest_old_sweep(log_dir):
    candidates = sorted(glob.glob(os.path.join(log_dir, "frozen-readout-sweep-seed-666-*")))
    return candidates[-1] if candidates else ""


def final_variant_for(setting_label):
    if setting_label == "loop1-stage-d":
        return "loop1-stage-d-plain-readout-final"
    return "stage-d-100k-plain-readout-final"


def baseline_variant_for(setting_label, family):
    if family == "cls":
        return "loop1-stage-d" if setting_label == "loop1-stage-d" else "stage-d-100k-control"
    if family == "adaptive":
        return "loop1-stage-d" if setting_label == "loop1-stage-d" else "stage-d-100k-control"
    if family == "mean":
        return "loop1-stage-d" if setting_label == "loop1-stage-d" else "stage-d-100k"
    raise ValueError(f"Unknown baseline family: {family}")


def baseline_strategy_for(setting_label, loop, family):
    if family == "cls":
        return "cls"
    if family == "mean":
        return "mean"
    if family == "adaptive":
        if setting_label == "loop1-stage-d":
            return "cls_tanh"
        if loop in {"loop-1", "loop-2"}:
            return "cls_tanh"
        return "loop_mean_cls"
    raise ValueError(f"Unknown baseline family: {family}")


def fixed_baseline_lr(setting_label, loop, family):
    if family == "cls":
        return "5e-5"
    if family == "adaptive" and setting_label != "loop1-stage-d":
        return "1e-3"
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--winner_json", required=True)
    parser.add_argument(
        "--results_root",
        default="./outputs/ft/rna-all/NoncodingRNAFamily/ecorna",
    )
    parser.add_argument(
        "--baseline_log_root",
        default="",
        help="Optional old frozen-readout sweep log root for validation-based mean and loop1 adaptive baseline selection.",
    )
    parser.add_argument("--output_json", default="")
    args = parser.parse_args()

    with open(args.winner_json, "r") as f:
        winner_payload = json.load(f)
    winner = winner_payload["winner"]

    results_root = os.path.abspath(args.results_root)
    logs_dir = os.path.abspath(
        os.path.join(results_root, "..", "logs")
    )
    baseline_log_root = args.baseline_log_root or maybe_latest_old_sweep(logs_dir)

    table_rows = []
    summary = {
        "winner": winner,
        "baseline_log_root": baseline_log_root,
        "settings": [],
    }

    for setting in SETTINGS:
        setting_label = setting["label"]
        loop = setting["loop"]
        loop_output = setting["loop_output"]
        final_variant = final_variant_for(setting_label)
        winner_paths = [
            (
                seed,
                build_result_path(
                    results_root,
                    final_variant,
                    winner["strategy"],
                    loop_output,
                    winner["lr"],
                    seed,
                ),
            )
            for seed in FINAL_SEEDS
        ]
        winner_metrics = aggregate_result_paths(winner_paths)

        baselines = {}
        for family in ("cls", "mean", "adaptive"):
            strategy = baseline_strategy_for(setting_label, loop, family)
            variant = baseline_variant_for(setting_label, family)
            lr = fixed_baseline_lr(setting_label, loop, family)

            if not lr:
                if baseline_log_root:
                    selected = select_from_logs(baseline_log_root, setting_label, loop, strategy)
                    lr = selected["lr"] if selected else ""
                else:
                    selected = None
            else:
                selected = None

            if not lr:
                baselines[family] = {
                    "variant": variant,
                    "strategy": strategy,
                    "lr": "",
                    "metrics": None,
                }
                continue

            candidate_seeds = FINAL_SEEDS if variant.endswith("-control") else FINAL_SEEDS
            result_paths = [
                (
                    seed,
                    build_result_path(
                        results_root,
                        variant,
                        strategy,
                        loop_output,
                        lr,
                        seed,
                    ),
                )
                for seed in candidate_seeds
            ]
            metrics = aggregate_result_paths(result_paths)
            baselines[family] = {
                "variant": variant,
                "strategy": strategy,
                "lr": lr,
                "selection": selected,
                "metrics": metrics,
            }

        setting_summary = {
            "setting": f"{setting_label}__{loop}",
            "winner": winner_metrics,
            "baselines": baselines,
        }
        summary["settings"].append(setting_summary)

        def fmt_metrics(item):
            metrics = item["metrics"] if item and "metrics" in item else item
            if metrics is None:
                return "NA", "NA", "0", "0", "0"
            return (
                f"{metrics['mean_test_accuracy']:.6f}",
                f"{metrics['mean_test_f1']:.6f}",
                str(metrics["count"]),
                str(metrics.get("content_mean_fallback_batches", 0)),
                str(metrics.get("content_mean_fallback_examples", 0)),
            )

        winner_acc, winner_f1, winner_count, winner_fb_batches, winner_fb_examples = fmt_metrics(winner_metrics)
        cls_acc, cls_f1, cls_count, _, _ = fmt_metrics(baselines["cls"])
        mean_acc, mean_f1, mean_count, _, _ = fmt_metrics(baselines["mean"])
        adaptive_acc, adaptive_f1, adaptive_count, _, _ = fmt_metrics(baselines["adaptive"])

        table_rows.append(
            {
                "setting": setting_summary["setting"],
                "winner_acc": winner_acc,
                "winner_f1": winner_f1,
                "winner_count": winner_count,
                "winner_fb_batches": winner_fb_batches,
                "winner_fb_examples": winner_fb_examples,
                "cls_acc": cls_acc,
                "cls_f1": cls_f1,
                "cls_count": cls_count,
                "mean_acc": mean_acc,
                "mean_f1": mean_f1,
                "mean_count": mean_count,
                "mean_lr": baselines["mean"]["lr"] or "NA",
                "adaptive_acc": adaptive_acc,
                "adaptive_f1": adaptive_f1,
                "adaptive_count": adaptive_count,
                "adaptive_lr": baselines["adaptive"]["lr"] or "NA",
            }
        )

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(summary, f, indent=2)

    print(
        "setting\twinner(strategy={strategy},lr={lr})\twinner_acc\twinner_f1\twinner_n\twinner_fb_batches\twinner_fb_examples\tcls_acc\tcls_f1\tcls_n\tmean_acc\tmean_f1\tmean_n\tmean_lr\tadaptive_acc\tadaptive_f1\tadaptive_n\tadaptive_lr".format(
            **winner
        )
    )
    for row in table_rows:
        print(
            "\t".join(
                [
                    row["setting"],
                    f"{winner['strategy']}/{winner['lr']}",
                    row["winner_acc"],
                    row["winner_f1"],
                    row["winner_count"],
                    row["winner_fb_batches"],
                    row["winner_fb_examples"],
                    row["cls_acc"],
                    row["cls_f1"],
                    row["cls_count"],
                    row["mean_acc"],
                    row["mean_f1"],
                    row["mean_count"],
                    row["mean_lr"],
                    row["adaptive_acc"],
                    row["adaptive_f1"],
                    row["adaptive_count"],
                    row["adaptive_lr"],
                ]
            )
        )


if __name__ == "__main__":
    main()
