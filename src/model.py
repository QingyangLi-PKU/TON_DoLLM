from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel


@dataclass
class DoLLMConfig:
    model_name_or_path: str = "bert-base-uncased"
    projection_output_dim: int = 256
    output_classes: int = 2
    num_flows: int = 64
    dropout: float = 0.0
    llm_num_hidden_layers: int | None = None
    torch_dtype: str = "fp32"


class FlowEncoder(nn.Module):
    def __init__(self, num_features: int, llm_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, llm_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DoLLMModel(nn.Module):
    def __init__(self, cfg: DoLLMConfig):
        super().__init__()
        base_cfg = AutoConfig.from_pretrained(cfg.model_name_or_path)
        if cfg.llm_num_hidden_layers is not None:
            base_cfg.num_hidden_layers = cfg.llm_num_hidden_layers

        self.llm = AutoModel.from_pretrained(cfg.model_name_or_path, config=base_cfg)
        for param in self.llm.parameters():
            param.requires_grad = False

        self.llm_dim = base_cfg.hidden_size
        self.flow_encoder = FlowEncoder(num_features=9, llm_dim=self.llm_dim, dropout=cfg.dropout)
        self.output_projection = nn.Linear(self.llm_dim, cfg.projection_output_dim)
        self.classifier = nn.Linear(cfg.projection_output_dim, cfg.output_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        flow_embeds = self.flow_encoder(inputs)
        transformer_out = self.llm(inputs_embeds=flow_embeds).last_hidden_state
        projected = self.output_projection(transformer_out.float())
        return self.classifier(projected)


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets_one_hot = F.one_hot(targets.long(), num_classes=2).float()
        bce = F.binary_cross_entropy_with_logits(logits, targets_one_hot, reduction="none")
        pt = torch.exp(-bce)
        focal = self.alpha * ((1 - pt) ** self.gamma) * bce
        return focal.sum(dim=1).mean()
