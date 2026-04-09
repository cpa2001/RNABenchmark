#!/usr/bin/env python

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import sklearn
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(REPO_ROOT, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ecorna import EcoRNAModel, EcoRNATokenizer


def load_split(data_dir: str, filename: str) -> Tuple[List[str], np.ndarray]:
    path = os.path.join(data_dir, filename)
    with open(path, "r") as f:
        rows = list(csv.reader(f))[1:]
    seqs = [row[0].upper().replace("U", "T") for row in rows]
    labels = np.asarray([int(row[1]) for row in rows], dtype=np.int64)
    return seqs, labels


def calculate_metric_with_sklearn(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": sklearn.metrics.accuracy_score(labels, predictions),
        "f1": sklearn.metrics.f1_score(labels, predictions, average="macro", zero_division=0),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(labels, predictions),
        "precision": sklearn.metrics.precision_score(labels, predictions, average="macro", zero_division=0),
        "recall": sklearn.metrics.recall_score(labels, predictions, average="macro", zero_division=0),
    }


@torch.no_grad()
def extract_features(model, tokenizer, sequences: List[str], batch_size: int, max_length: int, pooling: str, num_loops):
    device = next(model.parameters()).device
    outputs = []
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding="longest",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        result = model(**encoded, num_loops=num_loops)
        hidden = result.last_hidden_state
        attention_mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        if pooling == "cls":
            pooled = hidden[:, 0]
        elif pooling == "mean":
            pooled = (hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1e-6)
        else:
            raise ValueError(f"Unsupported prototype pooling: {pooling}")
        outputs.append(pooled.float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def nearest_centroid_predict(train_features: np.ndarray, train_labels: np.ndarray, query_features: np.ndarray):
    label_values = np.asarray(sorted(set(train_labels.tolist())), dtype=np.int64)
    centroids = []
    for label in label_values:
        centroids.append(train_features[train_labels == label].mean(axis=0))
    centroid_matrix = np.stack(centroids, axis=0)
    query_norm = (query_features ** 2).sum(axis=1, keepdims=True)
    centroid_norm = (centroid_matrix ** 2).sum(axis=1)[None, :]
    distances = query_norm + centroid_norm - 2 * query_features @ centroid_matrix.T
    return label_values[np.argmin(distances, axis=1)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--pooling", choices=["cls", "mean"], required=True)
    parser.add_argument("--num_loops", type=int, default=-1)
    parser.add_argument("--model_max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train_sequences, train_labels = load_split(args.data_dir, "train.csv")
    val_sequences, val_labels = load_split(args.data_dir, "val.csv")
    test_sequences, test_labels = load_split(args.data_dir, "test.csv")

    tokenizer = EcoRNATokenizer.from_pretrained(
        args.checkpoint_path,
        replace_u_with_t=True,
        model_max_length=args.model_max_length,
    )
    model = EcoRNAModel.from_pretrained(args.checkpoint_path).to(args.device)
    model.eval()

    num_loops = args.num_loops if args.num_loops > 0 else None

    train_features = extract_features(
        model, tokenizer, train_sequences, args.batch_size, args.model_max_length, args.pooling, num_loops
    )
    val_features = extract_features(
        model, tokenizer, val_sequences, args.batch_size, args.model_max_length, args.pooling, num_loops
    )
    test_features = extract_features(
        model, tokenizer, test_sequences, args.batch_size, args.model_max_length, args.pooling, num_loops
    )

    val_predictions = nearest_centroid_predict(train_features, train_labels, val_features)
    test_predictions = nearest_centroid_predict(train_features, train_labels, test_features)

    payload = {
        "checkpoint_path": args.checkpoint_path,
        "pooling": args.pooling,
        "num_loops": args.num_loops,
        "val": calculate_metric_with_sklearn(val_predictions, val_labels),
        "test": calculate_metric_with_sklearn(test_predictions, test_labels),
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
