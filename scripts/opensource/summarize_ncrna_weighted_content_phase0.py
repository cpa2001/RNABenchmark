#!/usr/bin/env python

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple


SEED = 666
VARIANT = "stage-d-100k-weighted-content-phase0"
LOOP_OUTPUT = "loops-3"
LRS = ["2e-4", "1e-3"]
STRATEGIES = ["weighted_layer_content", "weighted_cell_content"]


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


def build_run_dir(results_root: str, strategy: str, lr: str, seed: int) -> str:
    return os.path.join(
        results_root,
        VARIANT,
        strategy,
        LOOP_OUTPUT,
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


def candidate_sort_key(item: Dict[str, object]) -> Tuple[bool, float, float, int]:
    return (
        bool(item["passes_gate"]),
        float(item["best_val_f1"]) if item["best_val_f1"] is not None else -1.0,
        float(item["best_val_accuracy"]) if item["best_val_accuracy"] is not None else -1.0,
        1 if item["strategy"] == "weighted_layer_content" else 0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default="./outputs/ft/rna-all/NoncodingRNAFamily/ecorna")
    parser.add_argument("--output_json", default="")
    args = parser.parse_args()

    results_root = os.path.abspath(args.results_root)
    candidates: List[Dict[str, object]] = []

    for strategy in STRATEGIES:
        for lr in LRS:
            run_dir = build_run_dir(results_root, strategy, lr, SEED)
            result_path = build_result_path(run_dir, lr)
            state, best_eval = get_best_eval(run_dir)
            candidate = {
                "strategy": strategy,
                "lr": lr,
                "run_dir": run_dir,
                "complete": best_eval is not None and os.path.exists(result_path),
            }
            if not candidate["complete"]:
                candidate.update(
                    {
                        "best_val_accuracy": None,
                        "best_val_f1": None,
                        "best_epoch": None,
                        "test_accuracy": None,
                        "test_f1": None,
                        "pooler_mean_max_weight": None,
                        "pooler_effective_cells": None,
                        "pooler_max_abs_raw_logit": None,
                        "pooler_max_abs_bounded_logit": None,
                        "passes_gate": False,
                    }
                )
                candidates.append(candidate)
                continue

            metrics = load_json(result_path)
            candidate.update(
                {
                    "best_val_accuracy": best_eval["eval_accuracy"],
                    "best_val_f1": best_eval["eval_f1"],
                    "best_epoch": best_eval["epoch"],
                    "best_model_checkpoint": state.get("best_model_checkpoint"),
                    "test_accuracy": metrics["eval_accuracy"],
                    "test_f1": metrics["eval_f1"],
                    "pooler_mean_max_weight": metrics.get("pooler_mean_max_weight"),
                    "pooler_effective_cells": metrics.get("pooler_effective_cells"),
                    "pooler_max_abs_raw_logit": metrics.get("pooler_max_abs_raw_logit"),
                    "pooler_max_abs_bounded_logit": metrics.get("pooler_max_abs_bounded_logit"),
                }
            )
            candidate["passes_gate"] = (
                candidate["best_val_f1"] is not None
                and float(candidate["best_val_f1"]) >= 0.85
                and float(candidate["pooler_mean_max_weight"]) <= 0.75
                and float(candidate["pooler_effective_cells"]) >= 3.0
                and float(candidate["pooler_max_abs_raw_logit"]) <= 20.0
            )
            candidates.append(candidate)

    candidates.sort(key=candidate_sort_key, reverse=True)
    passing_strategies = sorted({item["strategy"] for item in candidates if item["passes_gate"]})
    winner = candidates[0] if candidates else None

    payload = {
        "results_root": results_root,
        "candidates": candidates,
        "passing_strategies": passing_strategies,
        "winner": winner,
        "pass_phase0": bool(passing_strategies),
        "next_action": "run_pilot" if passing_strategies else "stay_on_fixed_cell",
    }

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2)

    print("strategy\tlr\tbest_val_f1\tbest_val_acc\ttest_f1\tmax_w\teff_cells\tmax_raw_logit\tpass_phase0")
    for item in candidates:
        fields = [
            item["strategy"],
            item["lr"],
            "NA" if item["best_val_f1"] is None else f"{item['best_val_f1']:.6f}",
            "NA" if item["best_val_accuracy"] is None else f"{item['best_val_accuracy']:.6f}",
            "NA" if item["test_f1"] is None else f"{item['test_f1']:.6f}",
            "NA" if item["pooler_mean_max_weight"] is None else f"{item['pooler_mean_max_weight']:.6f}",
            "NA" if item["pooler_effective_cells"] is None else f"{item['pooler_effective_cells']:.6f}",
            "NA" if item["pooler_max_abs_raw_logit"] is None else f"{item['pooler_max_abs_raw_logit']:.6f}",
            str(item["passes_gate"]),
        ]
        print("\t".join(fields))

    if winner is None:
        raise SystemExit("No phase0 candidates found.")


if __name__ == "__main__":
    main()
