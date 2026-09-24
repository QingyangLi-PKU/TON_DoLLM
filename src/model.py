from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel


@dataclass
class DoLLMConfig:
    model_name_or_path: str = "/mnt/ssd1/model_zoo/roberta-base"
    projection_output_dim: int = 256
    output_classes: int = 2
    num_flows: int = 64
    dropout: float = 0.0
    llm_num_hidden_layers: int | None = None
    torch_dtype: str = "fp32"


class FlowEncoder(nn.Module):
    def __init__(self, num_features: int, llm_dim: int, dropout: float):
        super().__init__()
        # Keep the module nesting used by the original implementation.  Apart
        # from making checkpoints interchangeable, this preserves parameter
        # initialization order for exact seeded reproduction.
        self.model = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, llm_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class OutputProjection(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(input_dim, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Classifier(nn.Module):
    def __init__(self, projection_dim: int, output_classes: int):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(projection_dim, output_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DoLLMModel(nn.Module):
    def __init__(self, cfg: DoLLMConfig):
        super().__init__()
        self.llm_config = AutoConfig.from_pretrained(cfg.model_name_or_path)
        if cfg.llm_num_hidden_layers is not None:
            self.llm_config.num_hidden_layers = cfg.llm_num_hidden_layers

        self.llm = AutoModel.from_pretrained(
            cfg.model_name_or_path, config=self.llm_config
        )
        for param in self.llm.parameters():
            param.requires_grad = False

        self.llm_dim = self.llm_config.hidden_size
        # Names and construction order mirror dollm_whole/multi_inputs_model.py.
        self.flows_tokenizer = FlowEncoder(num_features=9, llm_dim=self.llm_dim, dropout=cfg.dropout)
        self.output_projection = OutputProjection(self.llm_dim, cfg.projection_output_dim)
        self.classifier = Classifier(cfg.projection_output_dim, cfg.output_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        flow_embeds = self.flows_tokenizer(inputs)
        transformer_out = self.llm(inputs_embeds=flow_embeds).last_hidden_state
        projected = self.output_projection(transformer_out)
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
