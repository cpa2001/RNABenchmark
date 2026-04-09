#!/usr/bin/env python

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple


SEED = 666
EXPECTED_SETTINGS = [
    ("loop1-stage-d", "loop-ckpt", "loops-ckpt", "loop1-stage-d-weighted-content-pilot"),
    ("stage-d-100k", "loop-1", "loops-1", "stage-d-100k-weighted-content-pilot"),
    ("stage-d-100k", "loop-2", "loops-2", "stage-d-100k-weighted-content-pilot"),
    ("stage-d-100k", "loop-3", "loops-3", "stage-d-100k-weighted-content-pilot"),
]
LRS = ["5e-5", "2e-4", "1e-3"]
STRATEGIES = ["weighted_layer_content", "weighted_cell_content"]
FIXED_CELL_WINNER_JSON = (
    "./outputs/ft/rna-all/NoncodingRNAFamily/logs/fixed-cell-pilot-winner-20260406_095456.json"
)


def load_json(path: str) -> Dict[str, object]:
    with open(path, "r") as f:
        return json.load(f)


def parse_step_from_checkpoint(path: Optional[str]) -> Optional[int]:
    if not path:
        return None
    base = os.path.basename(path.rstrip("/"))
    if not base.startswith("checkpoint-"):
        return None
    try:
        return int(base.split("-", 1)[1])
    except ValueError:
        return None


def build_run_dir(results_root: str, variant: str, strategy: str, loop_output: str, lr: str, seed: int) -> str:
    return os.path.join(
        results_root,
        variant,
        strategy,
        loop_output,
        "frozen-head",
        f"lr-{lr}",
        str(seed),
    )


def build_result_path(run_dir: str, lr: str) -> str:
    return os.path.join(
        run_dir,
        "results",
        f"ecorna_ncrna_frozen_head_lr-{lr}",
        "test_results.json",
    )


def get_best_eval(run_dir: str) -> Tuple[Optional[Dict[str, object]], Optional[Dict[str, object]]]:
    checkpoints = sorted(
        [
            os.path.join(run_dir, name, "trainer_state.json")
            for name in os.listdir(run_dir)
            if name.startswith("checkpoint-")
        ]
    ) if os.path.isdir(run_dir) else []
    if not checkpoints:
        return None, None
    state = load_json(checkpoints[-1])
    best_step = parse_step_from_checkpoint(state.get("best_model_checkpoint"))
    if best_step is None:
        return state, None
    for item in state.get("log_history", []):
        if item.get("step") == best_step and "eval_f1" in item:
            return state, item
    return state, None


def candidate_sort_key(item: Dict[str, object]) -> Tuple[bool, float, float, int, float]:
    return (
        bool(item["passes_gate"]),
        float(item["mean_val_f1"]) if item["mean_val_f1"] is not None else -1.0,
        float(item["mean_val_accuracy"]) if item["mean_val_accuracy"] is not None else -1.0,
        1 if item["strategy"] == "weighted_layer_content" else 0,
        -float(item["lr"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default="./outputs/ft/rna-all/NoncodingRNAFamily/ecorna")
    parser.add_argument("--baseline_winner_json", default=FIXED_CELL_WINNER_JSON)
    parser.add_argument("--output_json", default="")
    args = parser.parse_args()

    results_root = os.path.abspath(args.results_root)
    baseline_winner = load_json(args.baseline_winner_json)["winner"]
    baseline_mean_val_f1 = float(baseline_winner["mean_val_f1"])

    candidates: List[Dict[str, object]] = []
    for strategy in STRATEGIES:
        for lr in LRS:
            rows = []
            missing_settings = []
            for setting_label, loop_label, loop_output, variant in EXPECTED_SETTINGS:
                run_dir = build_run_dir(results_root, variant, strategy, loop_output, lr, SEED)
                result_path = build_result_path(run_dir, lr)
                state, best_eval = get_best_eval(run_dir)
                if best_eval is None or not os.path.exists(result_path):
                    missing_settings.append(f"{setting_label}__{loop_label}")
                    continue
                test_metrics = load_json(result_path)
                rows.append(
                    {
                        "setting": f"{setting_label}__{loop_label}",
                        "run_dir": run_dir,
                        "best_val_accuracy": best_eval["eval_accuracy"],
                        "best_val_f1": best_eval["eval_f1"],
                        "best_epoch": best_eval["epoch"],
                        "best_model_checkpoint": state.get("best_model_checkpoint") if state else None,
                        "test_accuracy": test_metrics["eval_accuracy"],
                        "test_f1": test_metrics["eval_f1"],
                        "pooler_mean_max_weight": test_metrics.get("pooler_mean_max_weight"),
                        "pooler_effective_cells": test_metrics.get("pooler_effective_cells"),
                        "pooler_max_abs_raw_logit": test_metrics.get("pooler_max_abs_raw_logit"),
                        "pooler_max_abs_bounded_logit": test_metrics.get("pooler_max_abs_bounded_logit"),
                        "pooler_mean_weights": test_metrics.get("pooler_mean_weights"),
                    }
                )

            candidate = {
                "strategy": strategy,
                "lr": lr,
                "num_settings": len(rows),
                "complete": len(rows) == len(EXPECTED_SETTINGS),
                "missing_settings": missing_settings,
                "rows": rows,
            }
            if rows:
                candidate["mean_val_accuracy"] = sum(row["best_val_accuracy"] for row in rows) / len(rows)
                candidate["mean_val_f1"] = sum(row["best_val_f1"] for row in rows) / len(rows)
            else:
                candidate["mean_val_accuracy"] = None
                candidate["mean_val_f1"] = None

            by_setting = {row["setting"]: row for row in rows}
            loop1 = by_setting.get("loop1-stage-d__loop-ckpt")
            candidate["passes_gate"] = (
                candidate["complete"]
                and float(candidate["mean_val_f1"]) >= (baseline_mean_val_f1 - 0.005)
                and all(
                    row.get("pooler_max_abs_raw_logit") is not None
                    and float(row["pooler_max_abs_raw_logit"]) <= 20.0
                    for row in rows
                )
                and loop1 is not None
                and float(loop1["pooler_mean_max_weight"]) <= 0.80
                and float(loop1["pooler_effective_cells"]) >= 2.0
            )
            candidates.append(candidate)

    candidates.sort(key=candidate_sort_key, reverse=True)
    winner = candidates[0] if candidates else None
    pass_pilot = bool(winner and winner["passes_gate"])

    payload = {
        "results_root": results_root,
        "baseline_winner_json": os.path.abspath(args.baseline_winner_json),
        "baseline_fixed_cell_mean_val_f1": baseline_mean_val_f1,
        "candidates": candidates,
        "winner": winner,
        "pass_pilot": pass_pilot,
        "next_action": "run_final" if pass_pilot else "stay_on_fixed_cell",
    }

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2)

    print("strategy\tlr\tsettings\tmean_val_f1\tmean_val_acc\tpass_pilot")
    for item in candidates:
        mean_f1 = "NA" if item["mean_val_f1"] is None else f"{item['mean_val_f1']:.6f}"
        mean_acc = "NA" if item["mean_val_accuracy"] is None else f"{item['mean_val_accuracy']:.6f}"
        print("\t".join([item["strategy"], item["lr"], str(item["num_settings"]), mean_f1, mean_acc, str(item["passes_gate"])]))

    if winner is None:
        raise SystemExit("No weighted-content pilot candidates found.")


if __name__ == "__main__":
    main()
