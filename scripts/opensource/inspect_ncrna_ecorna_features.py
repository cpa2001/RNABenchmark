#!/usr/bin/env python

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from contextlib import nullcontext
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BENCHMARK_ROOT.parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.ecorna import EcoRNAForSequenceClassification
from ecorna import EcoRNATokenizer


def load_split(data_dir: str, filename: str, limit: int = 0) -> Tuple[List[str], np.ndarray]:
    path = os.path.join(data_dir, filename)
    with open(path, "r") as f:
        rows = list(csv.reader(f))[1:]
    if limit > 0:
        rows = rows[:limit]
    seqs = [row[0].upper().replace("U", "T") for row in rows]
    labels = np.asarray([int(row[1]) for row in rows], dtype=np.int64)
    return seqs, labels


def select_balanced_subset(
    sequences: Sequence[str],
    labels: np.ndarray,
    limit: int,
    seed: int,
) -> Tuple[List[str], np.ndarray]:
    if limit <= 0 or limit >= len(labels):
        return list(sequences), labels

    rng = np.random.default_rng(seed)
    unique_labels = sorted(set(labels.tolist()))
    per_class = max(limit // len(unique_labels), 1)
    remainder = max(limit - per_class * len(unique_labels), 0)
    selected_indices: List[int] = []
    for offset, label in enumerate(unique_labels):
        candidates = np.where(labels == label)[0].copy()
        rng.shuffle(candidates)
        take = min(len(candidates), per_class + (1 if offset < remainder else 0))
        selected_indices.extend(candidates[:take].tolist())
    selected_indices = sorted(selected_indices)
    return [sequences[idx] for idx in selected_indices], labels[selected_indices]


def resolve_train_filename(data_dir: str) -> str:
    train_new = os.path.join(data_dir, "train_new.csv")
    if os.path.exists(train_new):
        return "train_new.csv"
    return "train.csv"


def calculate_metric_with_sklearn(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "matthews_correlation": float(matthews_corrcoef(labels, predictions)),
        "precision": float(precision_score(labels, predictions, average="macro", zero_division=0)),
        "recall": float(recall_score(labels, predictions, average="macro", zero_division=0)),
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def extract_feature_bundle(
    model: EcoRNAForSequenceClassification,
    tokenizer: EcoRNATokenizer,
    sequences: Sequence[str],
    batch_size: int,
    max_length: int,
    num_loops: int,
    device: str,
    use_bf16_autocast: bool,
) -> Tuple[Dict[str, np.ndarray], torch.Tensor]:
    features = {"cls": [], "mean": [], "loop_mean_cls": []}
    loop_cls_batches: List[torch.Tensor] = []
    for start in range(0, len(sequences), batch_size):
        batch = list(sequences[start : start + batch_size])
        encoded = tokenizer(
            batch,
            padding="longest",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_bf16_autocast and device.startswith("cuda")
            else nullcontext()
        )
        with autocast_ctx:
            outputs = model.ecorna(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                num_loops=num_loops,
                output_hidden_states=True,
            )
        hidden = outputs.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        per_loop_cls = torch.stack(
            [model.ecorna.final_norm(h)[:, 0] for h in outputs.hidden_states],
            dim=1,
        )
        features["cls"].append(hidden[:, 0].float().cpu().numpy())
        features["mean"].append(
            ((hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6))
            .float()
            .cpu()
            .numpy()
        )
        features["loop_mean_cls"].append(per_loop_cls.mean(dim=1).float().cpu().numpy())
        loop_cls_batches.append(per_loop_cls.float().cpu())
    return (
        {name: np.concatenate(values, axis=0) for name, values in features.items()},
        torch.cat(loop_cls_batches, dim=0),
    )


def build_content_mask(
    model: EcoRNAForSequenceClassification,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    if not hasattr(model, "_build_content_mask"):
        raise ValueError("EcoRNA benchmark wrapper does not expose _build_content_mask.")
    return model._build_content_mask(input_ids, attention_mask)


@torch.no_grad()
def extract_loop_layer_content_grid(
    model: EcoRNAForSequenceClassification,
    tokenizer: EcoRNATokenizer,
    sequences: Sequence[str],
    batch_size: int,
    max_length: int,
    num_loops: int,
    device: str,
    use_bf16_autocast: bool,
) -> Dict[str, np.ndarray]:
    features: Dict[str, List[np.ndarray]] = defaultdict(list)
    for start in range(0, len(sequences), batch_size):
        batch = list(sequences[start : start + batch_size])
        encoded = tokenizer(
            batch,
            padding="longest",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_bf16_autocast and device.startswith("cuda")
            else nullcontext()
        )
        with autocast_ctx:
            outputs = model.ecorna(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                num_loops=num_loops,
                output_loop_layer_hidden_states=True,
            )
        if outputs.loop_layer_hidden_states is None:
            raise ValueError("EcoRNA backbone did not return loop_layer_hidden_states.")
        content_mask = build_content_mask(model, encoded["input_ids"], encoded["attention_mask"])
        final_norm = model.ecorna.final_norm
        for loop_idx, per_layer_states in enumerate(outputs.loop_layer_hidden_states, start=1):
            for layer_idx, hidden_state in enumerate(per_layer_states, start=1):
                pooled = model._masked_mean(final_norm(hidden_state), content_mask)
                features[f"loop-{loop_idx}__layer-{layer_idx}"].append(pooled.float().cpu().numpy())
    return {name: np.concatenate(values, axis=0) for name, values in features.items()}


@torch.no_grad()
def extract_loop_layer_cls_grid(
    model: EcoRNAForSequenceClassification,
    tokenizer: EcoRNATokenizer,
    sequences: Sequence[str],
    batch_size: int,
    max_length: int,
    num_loops: int,
    device: str,
    use_bf16_autocast: bool,
) -> Dict[str, np.ndarray]:
    features: Dict[str, List[np.ndarray]] = defaultdict(list)
    for start in range(0, len(sequences), batch_size):
        batch = list(sequences[start : start + batch_size])
        encoded = tokenizer(
            batch,
            padding="longest",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_bf16_autocast and device.startswith("cuda")
            else nullcontext()
        )
        with autocast_ctx:
            outputs = model.ecorna(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                num_loops=num_loops,
                output_loop_layer_hidden_states=True,
            )
        if outputs.loop_layer_hidden_states is None:
            raise ValueError("EcoRNA backbone did not return loop_layer_hidden_states.")
        final_norm = model.ecorna.final_norm
        for loop_idx, per_layer_states in enumerate(outputs.loop_layer_hidden_states, start=1):
            for layer_idx, hidden_state in enumerate(per_layer_states, start=1):
                pooled = final_norm(hidden_state)[:, 0]
                features[f"loop-{loop_idx}__layer-{layer_idx}"].append(pooled.float().cpu().numpy())
    return {name: np.concatenate(values, axis=0) for name, values in features.items()}


def summarize_center_drift(
    left: torch.Tensor,
    right: torch.Tensor,
    labels: np.ndarray,
    left_loop: int,
    right_loop: int,
) -> Dict[str, float]:
    cosines: List[float] = []
    deltas: List[float] = []
    for label in sorted(set(labels.tolist())):
        mask = torch.from_numpy(labels == label)
        left_center = left[mask].mean(dim=0)
        right_center = right[mask].mean(dim=0)
        cosines.append(float(torch.nn.functional.cosine_similarity(left_center, right_center, dim=0).item()))
        deltas.append(float(torch.norm(right_center - left_center, p=2).item()))
    return {
        "left_loop": left_loop,
        "right_loop": right_loop,
        "cosine_mean": float(np.mean(cosines)),
        "cosine_std": float(np.std(cosines)),
        "delta_mean": float(np.mean(deltas)),
        "delta_std": float(np.std(deltas)),
    }


def compute_scatter_ratio(features: torch.Tensor, labels: np.ndarray) -> Dict[str, float]:
    global_center = features.mean(dim=0)
    between_sum = 0.0
    within_sum = 0.0
    total = int(features.shape[0])
    for label in sorted(set(labels.tolist())):
        mask = torch.from_numpy(labels == label)
        cls_features = features[mask]
        cls_center = cls_features.mean(dim=0)
        count = int(cls_features.shape[0])
        between_sum += count * float(torch.sum((cls_center - global_center) ** 2).item())
        within_sum += float(torch.sum((cls_features - cls_center) ** 2).item())
    between = between_sum / max(total, 1)
    within = within_sum / max(total, 1)
    ratio = between / max(within, 1e-12)
    return {
        "between_class_mean_sqdist": float(between),
        "within_class_mean_sqdist": float(within),
        "ratio": float(ratio),
    }


def summarize_loop_stats(loop_cls: torch.Tensor, labels: np.ndarray) -> Dict[str, object]:
    loops = int(loop_cls.shape[1])
    payload: Dict[str, object] = {
        "num_examples": int(loop_cls.shape[0]),
        "num_loops": loops,
        "loop_stats": [],
        "pairwise": [],
        "class_center_drift": [],
        "scatter_ratio": [],
    }

    for idx in range(loops):
        current = loop_cls[:, idx]
        norms = current.norm(dim=1)
        payload["loop_stats"].append(
            {
                "loop": idx + 1,
                "norm_mean": float(norms.mean().item()),
                "norm_std": float(norms.std().item()),
            }
        )
        scatter = compute_scatter_ratio(current, labels)
        scatter["loop"] = idx + 1
        payload["scatter_ratio"].append(scatter)

    for idx in range(loops - 1):
        left = loop_cls[:, idx]
        right = loop_cls[:, idx + 1]
        cosine = torch.nn.functional.cosine_similarity(left, right, dim=1)
        delta = (right - left).norm(dim=1)
        payload["pairwise"].append(
            {
                "left_loop": idx + 1,
                "right_loop": idx + 2,
                "cosine_mean": float(cosine.mean().item()),
                "cosine_std": float(cosine.std().item()),
                "delta_mean": float(delta.mean().item()),
            }
        )
        payload["class_center_drift"].append(
            summarize_center_drift(left, right, labels, idx + 1, idx + 2)
        )

    average_cls = loop_cls.mean(dim=1)
    last_cls = loop_cls[:, -1]
    avg_vs_last = torch.nn.functional.cosine_similarity(average_cls, last_cls, dim=1)
    payload["average_vs_last"] = {
        "cosine_mean": float(avg_vs_last.mean().item()),
        "cosine_std": float(avg_vs_last.std().item()),
        "delta_mean": float((average_cls - last_cls).norm(dim=1).mean().item()),
    }
    if loops > 1:
        payload["first_vs_last_center_drift"] = summarize_center_drift(
            loop_cls[:, 0], loop_cls[:, -1], labels, 1, loops
        )
    return payload


def build_sklearn_probe_metrics(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    c_grid: Sequence[float],
) -> Dict[str, object]:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    val_scaled = scaler.transform(val_features)
    test_scaled = scaler.transform(test_features)

    grid_payload: Dict[str, object] = {}
    best_key = None
    best_val = None
    for c_value in c_grid:
        clf = LogisticRegression(
            max_iter=5000,
            solver="lbfgs",
            C=float(c_value),
        )
        clf.fit(train_scaled, train_labels)
        val_predictions = clf.predict(val_scaled)
        test_predictions = clf.predict(test_scaled)
        current = {
            "val": calculate_metric_with_sklearn(val_predictions, val_labels),
            "test": calculate_metric_with_sklearn(test_predictions, test_labels),
        }
        key = f"C={c_value:g}"
        grid_payload[key] = current
        current_rank = (
            current["val"]["accuracy"],
            current["val"]["f1"],
        )
        if best_val is None or current_rank > best_val:
            best_val = current_rank
            best_key = key

    return {
        "standardized": True,
        "grid": grid_payload,
        "best": {
            "selected": best_key,
            **grid_payload[best_key],
        },
    }


class LinearProbeHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(inputs)


def evaluate_torch_probe(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(features.to(device))
        predictions = logits.argmax(dim=-1).cpu().numpy()
    return calculate_metric_with_sklearn(predictions, labels.cpu().numpy())


def build_torch_probe_metrics(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    learning_rates: Sequence[float],
    epochs: int,
    batch_size: int,
    seed: int,
    device: str,
) -> Dict[str, object]:
    train_x = torch.tensor(train_features, dtype=torch.float32)
    val_x = torch.tensor(val_features, dtype=torch.float32)
    test_x = torch.tensor(test_features, dtype=torch.float32)
    train_y = torch.tensor(train_labels, dtype=torch.long)
    val_y = torch.tensor(val_labels, dtype=torch.long)
    test_y = torch.tensor(test_labels, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True,
    )
    num_classes = int(train_y.max().item()) + 1
    input_dim = int(train_x.shape[1])
    torch_device = torch.device(device)
    grid_payload: Dict[str, object] = {}
    best_key = None
    best_val = None

    for learning_rate in learning_rates:
        set_seed(seed)
        probe = LinearProbeHead(input_dim=input_dim, num_classes=num_classes).to(torch_device)
        optimizer = torch.optim.AdamW(probe.parameters(), lr=float(learning_rate), weight_decay=0.01)
        loss_fn = nn.CrossEntropyLoss()
        best_epoch = 0
        best_state = None
        best_metrics = None
        for epoch in range(1, epochs + 1):
            probe.train()
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(torch_device)
                batch_labels = batch_labels.to(torch_device)
                optimizer.zero_grad(set_to_none=True)
                logits = probe(batch_features)
                loss = loss_fn(logits, batch_labels)
                loss.backward()
                optimizer.step()
            val_metrics = evaluate_torch_probe(probe, val_x, val_y, torch_device)
            rank = (val_metrics["accuracy"], val_metrics["f1"])
            if best_val is None:
                pass
            if best_metrics is None or rank > (best_metrics["accuracy"], best_metrics["f1"]):
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in probe.state_dict().items()}
                best_metrics = val_metrics

        assert best_state is not None
        probe.load_state_dict(best_state)
        probe.to(torch_device)
        current = {
            "best_epoch": best_epoch,
            "val": best_metrics,
            "test": evaluate_torch_probe(probe, test_x, test_y, torch_device),
        }
        key = f"lr={learning_rate:g}"
        grid_payload[key] = current
        current_rank = (
            current["val"]["accuracy"],
            current["val"]["f1"],
        )
        if best_val is None or current_rank > best_val:
            best_val = current_rank
            best_key = key

    return {
        "standardized": False,
        "optimizer": "AdamW",
        "epochs": epochs,
        "batch_size": batch_size,
        "grid": grid_payload,
        "best": {
            "selected": best_key,
            **grid_payload[best_key],
        },
    }


def load_reference_results(
    results_root: str,
    loops: Sequence[int],
) -> Dict[str, object]:
    variants = {
        "cls": "lr-5e-5",
        "cls_tanh": "lr-1e-3",
        "loop_mean_cls": "lr-1e-3",
    }
    seeds = ["3407", "42", "666"]
    payload: Dict[str, object] = {}
    root = Path(results_root)
    for variant, lr in variants.items():
        payload[variant] = {}
        for num_loops in loops:
            accuracies = []
            f1_values = []
            for seed in seeds:
                result_path = (
                    root
                    / variant
                    / f"loops-{num_loops}"
                    / "frozen-head"
                    / lr
                    / seed
                    / "results"
                    / f"ecorna_ncrna_frozen_head_{lr}"
                    / "test_results.json"
                )
                if not result_path.exists():
                    continue
                metrics = json.loads(result_path.read_text())
                accuracies.append(float(metrics["eval_accuracy"]))
                f1_values.append(float(metrics["eval_f1"]))
            if accuracies:
                payload[variant][f"loops-{num_loops}"] = {
                    "acc_mean": float(np.mean(accuracies)),
                    "f1_mean": float(np.mean(f1_values)),
                    "acc_values": accuracies,
                }
    return payload


def build_attribution(
    reference_results: Dict[str, object],
    probe_results: Dict[str, object],
    representation_stats: Dict[str, object],
) -> Dict[str, object]:
    attribution: Dict[str, object] = {}
    plain_cls_reference = []
    plain_cls_probe = []
    scatter_ratio = []
    for loop_key in sorted(representation_stats.keys()):
        if "cls" in reference_results and loop_key in reference_results["cls"]:
            plain_cls_reference.append(reference_results["cls"][loop_key]["acc_mean"])
        if loop_key in probe_results and "cls" in probe_results[loop_key]:
            plain_cls_probe.append(probe_results[loop_key]["cls"]["torch_probe"]["best"]["test"]["accuracy"])
        scatter_items = representation_stats[loop_key]["scatter_ratio"]
        if scatter_items:
            scatter_ratio.append(scatter_items[-1]["ratio"])

    attribution["plain_cls_reference_mean_acc"] = plain_cls_reference
    attribution["plain_cls_torch_probe_acc"] = plain_cls_probe
    attribution["last_loop_scatter_ratio"] = scatter_ratio
    attribution["plain_cls_probe_matches_reference_drop"] = (
        len(plain_cls_reference) == len(plain_cls_probe)
        and len(plain_cls_reference) >= 2
        and plain_cls_probe[-1] < plain_cls_probe[0]
        and plain_cls_reference[-1] < plain_cls_reference[0]
    )
    attribution["scatter_ratio_drops_with_loops"] = (
        len(scatter_ratio) >= 2 and scatter_ratio[-1] < scatter_ratio[0]
    )
    attribution["suggested_root_cause"] = (
        "representation_drift"
        if attribution["plain_cls_probe_matches_reference_drop"]
        and attribution["scatter_ratio_drops_with_loops"]
        else "diagnostic_path_mismatch_or_inconclusive"
    )
    return attribution


def cell_sort_key(cell_key: str) -> Tuple[int, int]:
    left, right = cell_key.split("__")
    return int(left.replace("loop-", "")), int(right.replace("layer-", ""))


def build_loop_layer_heatmap(cells: Dict[str, object]) -> Dict[str, object]:
    if not cells:
        return {}

    loops = max(cell_sort_key(cell_key)[0] for cell_key in cells)
    layers = max(cell_sort_key(cell_key)[1] for cell_key in cells)

    def empty_grid():
        return [[None for _ in range(layers)] for _ in range(loops)]

    heatmap = {
        "val_accuracy": empty_grid(),
        "val_f1": empty_grid(),
        "test_accuracy": empty_grid(),
        "test_f1": empty_grid(),
    }
    best_cell = None
    best_rank = None

    for cell_key, payload in cells.items():
        loop_idx, layer_idx = cell_sort_key(cell_key)
        best = payload["sklearn_probe"]["best"]
        heatmap["val_accuracy"][loop_idx - 1][layer_idx - 1] = best["val"]["accuracy"]
        heatmap["val_f1"][loop_idx - 1][layer_idx - 1] = best["val"]["f1"]
        heatmap["test_accuracy"][loop_idx - 1][layer_idx - 1] = best["test"]["accuracy"]
        heatmap["test_f1"][loop_idx - 1][layer_idx - 1] = best["test"]["f1"]

        rank = (best["val"]["f1"], best["val"]["accuracy"], -loop_idx, -layer_idx)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_cell = {
                "cell": cell_key,
                "loop": loop_idx,
                "layer": layer_idx,
                "val": best["val"],
                "test": best["test"],
            }

    heatmap["best_cell"] = best_cell
    return heatmap


def cell_key_to_cli_spec(cell_key: str) -> str:
    return cell_key.replace("__", ":")


def build_fixed_cell_manifest(
    family: str,
    loop_layer_payload: Dict[str, object],
    top_k: int,
) -> Dict[str, object]:
    ranking_payload: List[Dict[str, object]] = []
    scores: Dict[str, Dict[str, object]] = defaultdict(
        lambda: {
            "val_f1": [],
            "val_accuracy": [],
            "loop_keys": [],
        }
    )

    for loop_key, payload in loop_layer_payload.items():
        for cell_key, cell_payload in payload["cells"].items():
            best = cell_payload["sklearn_probe"]["best"]
            scores[cell_key]["val_f1"].append(float(best["val"]["f1"]))
            scores[cell_key]["val_accuracy"].append(float(best["val"]["accuracy"]))
            scores[cell_key]["loop_keys"].append(loop_key)

    for cell_key, values in scores.items():
        loop_idx, layer_idx = cell_sort_key(cell_key)
        ranking_payload.append(
            {
                "cell": cell_key,
                "cell_spec": cell_key_to_cli_spec(cell_key),
                "loop": loop_idx,
                "layer": layer_idx,
                "mean_val_f1": float(np.mean(values["val_f1"])),
                "mean_val_accuracy": float(np.mean(values["val_accuracy"])),
                "count": int(len(values["val_f1"])),
                "loop_keys": sorted(values["loop_keys"]),
            }
        )

    ranking_payload.sort(
        key=lambda item: (
            item["mean_val_f1"],
            item["mean_val_accuracy"],
            -item["loop"],
            -item["layer"],
        ),
        reverse=True,
    )

    selected = ranking_payload[:top_k]
    selected_cells = [item["cell_spec"] for item in selected]
    selected_joined = ",".join(selected_cells)
    selected_hash = hashlib.sha1(selected_joined.encode("utf-8")).hexdigest()[:8] if selected_cells else ""

    return {
        "family": family,
        "top_k": top_k,
        "selected_cells": selected_cells,
        "selected_hash": selected_hash,
        "selected_tag": f"{family}-{selected_hash}" if selected_hash else family,
        "ranking": ranking_payload,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--loops", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--modes", nargs="+", default=["cls", "mean", "loop_mean_cls"])
    parser.add_argument("--model_max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--stats_limit", type=int, default=512)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--disable_bf16_autocast", action="store_true")
    parser.add_argument("--sklearn_c_grid", nargs="+", type=float, default=[0.1, 1.0, 10.0, 100.0])
    parser.add_argument("--torch_probe_lrs", nargs="+", type=float, default=[5e-5, 1e-3])
    parser.add_argument("--torch_probe_epochs", type=int, default=30)
    parser.add_argument("--torch_probe_batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reference_results_root", default="")
    parser.add_argument("--include_loop_layer_content_grid", action="store_true")
    parser.add_argument("--include_loop_layer_cls_grid", action="store_true")
    parser.add_argument("--emit_fixed_cell_manifest", action="store_true")
    parser.add_argument("--fixed_cell_top_k", type=int, default=3)
    args = parser.parse_args()

    set_seed(args.seed)
    train_filename = resolve_train_filename(args.data_dir)
    train_sequences, train_labels = load_split(args.data_dir, train_filename)
    val_sequences, val_labels = load_split(args.data_dir, "val.csv")
    test_sequences, test_labels = load_split(args.data_dir, "test.csv")
    stats_sequences, stats_labels = load_split(args.data_dir, "test.csv")
    stats_sequences, stats_labels = select_balanced_subset(
        sequences=stats_sequences,
        labels=stats_labels,
        limit=args.stats_limit,
        seed=args.seed,
    )

    tokenizer = EcoRNATokenizer(
        replace_u_with_t=True,
        model_max_length=args.model_max_length,
    )
    model = EcoRNAForSequenceClassification.from_pretrained(
        args.checkpoint_path,
        num_labels=len(set(train_labels.tolist())),
        problem_type="single_label_classification",
        token_type="single",
        pooling_strategy="cls",
        num_loops=None,
    ).to(args.device)
    model.eval()
    use_bf16_autocast = not args.disable_bf16_autocast

    payload: Dict[str, object] = {
        "checkpoint_path": args.checkpoint_path,
        "train_filename": train_filename,
        "modes": args.modes,
        "loops": args.loops,
        "note": (
            "Diagnostics now use the benchmark wrapper path (EcoRNAForSequenceClassification.ecorna) "
            "to match downstream evaluation. sklearn probes are standardized before fitting. "
            "Feature extraction uses bf16 autocast by default on CUDA to match the benchmark path. "
            "Torch probes train a plain linear head on cached frozen features with AdamW."
        ),
        "representation_stats": {},
        "probes": {},
    }
    include_content_grid = args.include_loop_layer_content_grid or args.emit_fixed_cell_manifest
    include_cls_grid = args.include_loop_layer_cls_grid or args.emit_fixed_cell_manifest

    if include_content_grid:
        payload["loop_layer_content_grid"] = {}
    if include_cls_grid:
        payload["loop_layer_cls_grid"] = {}

    for num_loops in args.loops:
        loop_key = f"loops-{num_loops}"
        stats_bundle, stats_loop_cls = extract_feature_bundle(
            model=model,
            tokenizer=tokenizer,
            sequences=stats_sequences,
            batch_size=args.batch_size,
            max_length=args.model_max_length,
            num_loops=num_loops,
            device=args.device,
            use_bf16_autocast=use_bf16_autocast,
        )
        payload["representation_stats"][loop_key] = summarize_loop_stats(stats_loop_cls, stats_labels)
        payload["probes"][loop_key] = {}

        train_bundle, _ = extract_feature_bundle(
            model=model,
            tokenizer=tokenizer,
            sequences=train_sequences,
            batch_size=args.batch_size,
            max_length=args.model_max_length,
            num_loops=num_loops,
            device=args.device,
            use_bf16_autocast=use_bf16_autocast,
        )
        val_bundle, _ = extract_feature_bundle(
            model=model,
            tokenizer=tokenizer,
            sequences=val_sequences,
            batch_size=args.batch_size,
            max_length=args.model_max_length,
            num_loops=num_loops,
            device=args.device,
            use_bf16_autocast=use_bf16_autocast,
        )
        test_bundle, _ = extract_feature_bundle(
            model=model,
            tokenizer=tokenizer,
            sequences=test_sequences,
            batch_size=args.batch_size,
            max_length=args.model_max_length,
            num_loops=num_loops,
            device=args.device,
            use_bf16_autocast=use_bf16_autocast,
        )

        for mode in args.modes:
            payload["probes"][loop_key][mode] = {
                "sklearn_probe": build_sklearn_probe_metrics(
                    train_features=train_bundle[mode],
                    train_labels=train_labels,
                    val_features=val_bundle[mode],
                    val_labels=val_labels,
                    test_features=test_bundle[mode],
                    test_labels=test_labels,
                    c_grid=args.sklearn_c_grid,
                ),
                "torch_probe": build_torch_probe_metrics(
                    train_features=train_bundle[mode],
                    train_labels=train_labels,
                    val_features=val_bundle[mode],
                    val_labels=val_labels,
                    test_features=test_bundle[mode],
                    test_labels=test_labels,
                    learning_rates=args.torch_probe_lrs,
                    epochs=args.torch_probe_epochs,
                    batch_size=args.torch_probe_batch_size,
                    seed=args.seed,
                    device="cpu",
                ),
            }

        if include_content_grid:
            train_grid = extract_loop_layer_content_grid(
                model=model,
                tokenizer=tokenizer,
                sequences=train_sequences,
                batch_size=args.batch_size,
                max_length=args.model_max_length,
                num_loops=num_loops,
                device=args.device,
                use_bf16_autocast=use_bf16_autocast,
            )
            val_grid = extract_loop_layer_content_grid(
                model=model,
                tokenizer=tokenizer,
                sequences=val_sequences,
                batch_size=args.batch_size,
                max_length=args.model_max_length,
                num_loops=num_loops,
                device=args.device,
                use_bf16_autocast=use_bf16_autocast,
            )
            test_grid = extract_loop_layer_content_grid(
                model=model,
                tokenizer=tokenizer,
                sequences=test_sequences,
                batch_size=args.batch_size,
                max_length=args.model_max_length,
                num_loops=num_loops,
                device=args.device,
                use_bf16_autocast=use_bf16_autocast,
            )
            cell_payload = {}
            for cell_key in sorted(train_grid.keys(), key=cell_sort_key):
                cell_payload[cell_key] = {
                    "sklearn_probe": build_sklearn_probe_metrics(
                        train_features=train_grid[cell_key],
                        train_labels=train_labels,
                        val_features=val_grid[cell_key],
                        val_labels=val_labels,
                        test_features=test_grid[cell_key],
                        test_labels=test_labels,
                        c_grid=args.sklearn_c_grid,
                    )
                }
            payload["loop_layer_content_grid"][loop_key] = {
                "cells": cell_payload,
                "heatmap": build_loop_layer_heatmap(cell_payload),
            }

        if include_cls_grid:
            train_cls_grid = extract_loop_layer_cls_grid(
                model=model,
                tokenizer=tokenizer,
                sequences=train_sequences,
                batch_size=args.batch_size,
                max_length=args.model_max_length,
                num_loops=num_loops,
                device=args.device,
                use_bf16_autocast=use_bf16_autocast,
            )
            val_cls_grid = extract_loop_layer_cls_grid(
                model=model,
                tokenizer=tokenizer,
                sequences=val_sequences,
                batch_size=args.batch_size,
                max_length=args.model_max_length,
                num_loops=num_loops,
                device=args.device,
                use_bf16_autocast=use_bf16_autocast,
            )
            test_cls_grid = extract_loop_layer_cls_grid(
                model=model,
                tokenizer=tokenizer,
                sequences=test_sequences,
                batch_size=args.batch_size,
                max_length=args.model_max_length,
                num_loops=num_loops,
                device=args.device,
                use_bf16_autocast=use_bf16_autocast,
            )
            cls_cell_payload = {}
            for cell_key in sorted(train_cls_grid.keys(), key=cell_sort_key):
                cls_cell_payload[cell_key] = {
                    "sklearn_probe": build_sklearn_probe_metrics(
                        train_features=train_cls_grid[cell_key],
                        train_labels=train_labels,
                        val_features=val_cls_grid[cell_key],
                        val_labels=val_labels,
                        test_features=test_cls_grid[cell_key],
                        test_labels=test_labels,
                        c_grid=args.sklearn_c_grid,
                    )
                }
            payload["loop_layer_cls_grid"][loop_key] = {
                "cells": cls_cell_payload,
                "heatmap": build_loop_layer_heatmap(cls_cell_payload),
            }

    if args.reference_results_root:
        payload["reference_results"] = load_reference_results(args.reference_results_root, args.loops)
        payload["attribution"] = build_attribution(
            reference_results=payload["reference_results"],
            probe_results=payload["probes"],
            representation_stats=payload["representation_stats"],
        )

    if args.emit_fixed_cell_manifest:
        manifest = {}
        if include_content_grid:
            manifest["fixed_cell_content"] = build_fixed_cell_manifest(
                family="fixed_cell_content",
                loop_layer_payload=payload["loop_layer_content_grid"],
                top_k=args.fixed_cell_top_k,
            )
        if include_cls_grid:
            manifest["fixed_cell_cls"] = build_fixed_cell_manifest(
                family="fixed_cell_cls",
                loop_layer_payload=payload["loop_layer_cls_grid"],
                top_k=args.fixed_cell_top_k,
            )
        payload["fixed_cell_manifest"] = manifest

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
