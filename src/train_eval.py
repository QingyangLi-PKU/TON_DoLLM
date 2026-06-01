from __future__ import annotations

import copy
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from .data import FlowEvalDataset, FlowTrainDataset, load_flow_csv, preprocess_flows
from .model import DoLLMConfig, DoLLMModel, FocalLoss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _predict(model: DoLLMModel, loader: DataLoader, current_detection_batch_size: int, device: torch.device):
    model.eval()
    preds = [-1 for _ in range(current_detection_batch_size)]
    labels = [-1 for _ in range(current_detection_batch_size)]
    probs = [-1.0 for _ in range(current_detection_batch_size)]

    with torch.no_grad():
        for x, y, flow_ids in loader:
            x = x.to(device)
            outputs = model(x)
            batch_probs = torch.softmax(outputs, dim=-1)[:, :, 1]
            batch_preds = torch.argmax(outputs, dim=-1)

            for p, t, fid, score in zip(
                batch_preds.reshape(-1).cpu().tolist(),
                y.reshape(-1).cpu().tolist(),
                flow_ids.reshape(-1).cpu().tolist(),
                batch_probs.reshape(-1).cpu().tolist(),
            ):
                if preds[fid] == -1:
                    preds[fid] = p
                    labels[fid] = t
                    probs[fid] = score
    return preds, labels, probs


def _metrics(y_pred, y_true, y_prob) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_prob)),
    }


def _build_loaders(
    train_csv: str,
    valid_csv: str,
    current_detection_batch_csv: str,
    num_flows: int,
    num_training_samples: int,
    batch_size: int,
    num_workers: int,
):
    train_records = preprocess_flows(load_flow_csv(train_csv))
    valid_records = preprocess_flows(load_flow_csv(valid_csv))
    current_detection_batch = preprocess_flows(load_flow_csv(current_detection_batch_csv))

    train_loader = DataLoader(
        FlowTrainDataset(train_records, num_flows=num_flows, num_training_samples=num_training_samples),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    valid_loader = DataLoader(
        FlowEvalDataset(valid_records, num_flows=num_flows),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    current_detection_batch_loader = DataLoader(
        FlowEvalDataset(current_detection_batch, num_flows=num_flows),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, valid_loader, current_detection_batch_loader, len(valid_records), len(current_detection_batch)


def train_best_model(
    cfg: DoLLMConfig,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    valid_size: int,
    device: torch.device,
    num_epochs: int = 5,
    learning_rate: float = 1e-4,
):
    model = DoLLMModel(cfg).to(device)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_f1 = -1.0
    best_state = None
    for epoch in range(num_epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x).view(-1, 2)
            labels = y.view(-1)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        y_pred, y_true, y_prob = _predict(model, valid_loader, valid_size, device)
        current_f1 = _metrics(y_pred, y_true, y_prob)["f1"]
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_state = copy.deepcopy(model.state_dict())
        print(f"[train] epoch={epoch + 1}/{num_epochs}, valid_f1={current_f1:.4f}")

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, best_f1


def evaluate_model(
    model: DoLLMModel,
    current_detection_batch_loader: DataLoader,
    current_detection_batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    y_pred, y_true, y_prob = _predict(model, current_detection_batch_loader, current_detection_batch_size, device)
    return _metrics(y_pred, y_true, y_prob)


def run_in_domain_experiment(
    dataset_dir: str,
    model_name_or_path: str,
    output_dir: str,
    seed: int = 42,
    num_flows: int = 64,
    num_training_samples: int = 15000,
    batch_size: int = 64,
    num_workers: int = 4,
    num_epochs: int = 20,
    learning_rate: float = 1e-4,
) -> Dict[str, float]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = Path(dataset_dir)
    train_csv = str(dataset_path / "mixed_flows_train.csv")
    valid_csv = str(dataset_path / "mixed_flows_valid.csv")
    current_detection_batch_csv = str(dataset_path / "mixed_flows_current_detection_batch.csv")

    train_loader, valid_loader, current_detection_batch_loader, valid_size, current_detection_batch_size = _build_loaders(
        train_csv,
        valid_csv,
        current_detection_batch_csv,
        num_flows,
        num_training_samples,
        batch_size,
        num_workers,
    )

    cfg = DoLLMConfig(
        model_name_or_path=model_name_or_path,
        projection_output_dim=256,
        output_classes=2,
        num_flows=num_flows,
        dropout=0.0,
    )
    model, best_valid_f1 = train_best_model(
        cfg=cfg,
        train_loader=train_loader,
        valid_loader=valid_loader,
        valid_size=valid_size,
        device=device,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
    )
    scores = evaluate_model(model, current_detection_batch_loader, current_detection_batch_size, device)
    scores["best_valid_f1"] = float(best_valid_f1)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(Path(output_dir) / "best_model.pt"))
    np.save(str(Path(output_dir) / "in_domain_metrics.npy"), scores, allow_pickle=True)
    return scores


def run_zero_shot_experiment(
    syn_dataset_dir: str,
    udp_dataset_dir: str,
    model_name_or_path: str,
    output_dir: str,
    seeds: Iterable[int] = (42, 123, 456),
    num_flows: int = 64,
    num_training_samples: int = 15000,
    batch_size: int = 64,
    num_workers: int = 4,
    num_epochs: int = 20,
    learning_rate: float = 1e-4,
) -> List[Dict[str, float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_pairs = [("SYN", syn_dataset_dir), ("UDP", udp_dataset_dir)]

    def build_bundle(base_dir: str):
        path = Path(base_dir)
        return _build_loaders(
            str(path / "mixed_flows_train.csv"),
            str(path / "mixed_flows_valid.csv"),
            str(path / "mixed_flows_current_detection_batch.csv"),
            num_flows,
            num_training_samples,
            batch_size,
            num_workers,
        )

    bundles = {name: build_bundle(ds_dir) for name, ds_dir in dataset_pairs}
    rows: List[Dict[str, float]] = []

    for seed in seeds:
        set_seed(int(seed))
        for src_name, _ in dataset_pairs:
            train_loader, valid_loader, _, valid_size, _ = bundles[src_name]
            cfg = DoLLMConfig(model_name_or_path=model_name_or_path, num_flows=num_flows)
            model, best_valid_f1 = train_best_model(
                cfg=cfg,
                train_loader=train_loader,
                valid_loader=valid_loader,
                valid_size=valid_size,
                device=device,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
            )

            for tgt_name, _ in dataset_pairs:
                _, _, current_detection_batch_loader, _, current_detection_batch_size = bundles[tgt_name]
                metrics = evaluate_model(model, current_detection_batch_loader, current_detection_batch_size, device)
                rows.append(
                    {
                        "seed": int(seed),
                        "source_dataset": src_name,
                        "target_dataset": tgt_name,
                        "setting": "in_domain" if src_name == tgt_name else "zero_shot",
                        "best_valid_f1": float(best_valid_f1),
                        **metrics,
                    }
                )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(output / "zero_shot_results.csv", index=False)
    zero_shot_df = df[df["setting"] == "zero_shot"]
    if not zero_shot_df.empty:
        summary = zero_shot_df[["f1", "accuracy", "precision", "recall", "auc"]].agg(["mean", "std"])
        summary.to_csv(output / "zero_shot_summary.csv")
    return rows
