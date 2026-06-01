from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

FEATURE_COLUMNS = [
    "Source Port",
    "Destination Port",
    "Protocol",
    "Total Packets",
    "Total Length of Packets",
    "Packet Length Max",
    "Packet Length Min",
    "Packet Length Mean",
    "Packet Length Std",
]

SORT_KEY_COLUMNS = [
    "Packet Length Mean",
    "Total Packets",
    "Protocol",
    "Source Port",
    "Destination Port",
]

CATEGORICAL_COLUMNS = ["Source Port", "Destination Port", "Protocol"]


@dataclass
class FlowRecord:
    features: Dict[str, float]
    binary_label: int
    flow_id: int


def load_flow_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def preprocess_flows(df: pd.DataFrame) -> List[FlowRecord]:
    working = df.copy()
    working["Binary_Label"] = (working["Label"] != "BENIGN").astype(int)
    working["Flow_ID"] = np.arange(len(working))

    for column in CATEGORICAL_COLUMNS:
        freqs = working[column].value_counts(normalize=True).to_dict()
        working[column] = working[column].map(freqs).fillna(0.0)

    scaler = StandardScaler()
    working[FEATURE_COLUMNS] = scaler.fit_transform(working[FEATURE_COLUMNS])
    working = working.sort_values(SORT_KEY_COLUMNS).reset_index(drop=True)

    records: List[FlowRecord] = []
    for _, row in working.iterrows():
        records.append(
            FlowRecord(
                features={name: float(row[name]) for name in FEATURE_COLUMNS},
                binary_label=int(row["Binary_Label"]),
                flow_id=int(row["Flow_ID"]),
            )
        )
    return records


def _binning_list(items: Sequence[FlowRecord], bins: int) -> List[List[FlowRecord]]:
    avg = len(items) // bins
    remainder = len(items) % bins
    result: List[List[FlowRecord]] = []
    start = 0
    for i in range(bins):
        end = start + avg + (1 if i < remainder else 0)
        result.append(list(items[start:end]))
        start = end
    return result


def make_train_sequences(
    records: Sequence[FlowRecord], num_flows: int, num_training_samples: int
) -> List[List[FlowRecord]]:
    binnings = _binning_list(records, num_flows)
    seqs: List[List[FlowRecord]] = []
    first_bin_len = len(binnings[0])

    for flow_idx in range(first_bin_len):
        seq = []
        for bin_items in binnings:
            seq.append(bin_items[flow_idx] if flow_idx < len(bin_items) else bin_items[-1])
        seqs.append(seq)

    samples_to_add = max(0, num_training_samples - len(seqs))
    for _ in range(samples_to_add):
        seq = [bin_items[np.random.randint(0, len(bin_items))] for bin_items in binnings]
        seqs.append(seq)
    return seqs


def make_eval_sequences(records: Sequence[FlowRecord], num_flows: int) -> List[List[FlowRecord]]:
    binnings = _binning_list(records, num_flows)
    seqs: List[List[FlowRecord]] = []
    first_bin_len = len(binnings[0])

    for flow_idx in range(first_bin_len):
        seq = []
        for bin_items in binnings:
            seq.append(bin_items[flow_idx] if flow_idx < len(bin_items) else bin_items[-1])
        seqs.append(seq)
    return seqs


class FlowTrainDataset(Dataset):
    def __init__(self, records: Sequence[FlowRecord], num_flows: int, num_training_samples: int):
        self.sequences = make_train_sequences(records, num_flows, num_training_samples)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        sequence = self.sequences[idx]
        features = torch.tensor(
            [[item.features[col] for col in FEATURE_COLUMNS] for item in sequence],
            dtype=torch.float32,
        )
        labels = torch.tensor([item.binary_label for item in sequence], dtype=torch.long)
        return features, labels


class FlowEvalDataset(Dataset):
    def __init__(self, records: Sequence[FlowRecord], num_flows: int):
        self.sequences = make_eval_sequences(records, num_flows)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        sequence = self.sequences[idx]
        features = torch.tensor(
            [[item.features[col] for col in FEATURE_COLUMNS] for item in sequence],
            dtype=torch.float32,
        )
        labels = torch.tensor([item.binary_label for item in sequence], dtype=torch.long)
        flow_ids = torch.tensor([item.flow_id for item in sequence], dtype=torch.long)
        return features, labels, flow_ids
