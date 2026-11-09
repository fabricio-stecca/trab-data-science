"""Construção de otimizador, scheduler e função de perda a partir da config."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.runner import resolve_weight_bias


# ----------------------------------------------------------------- otimizador
def build_optimizer(params, cfg: dict) -> torch.optim.Optimizer:
    name = cfg["name"].lower()
    lr, wd = cfg["lr"], cfg.get("weight_decay", 0.0)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    raise ValueError(f"otimizador '{name}' inválido (adam | adamw)")


# ----------------------------------------------------------------- scheduler
def build_scheduler(optimizer, cfg: dict, epochs: int):
    """Scheduler atualizado uma vez por época."""
    name = (cfg or {}).get("name", "none").lower()
    if name == "none":
        return None
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg["step_size"], gamma=cfg["gamma"])
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    raise ValueError(f"scheduler '{name}' inválido (none | step | cosine)")


# ----------------------------------------------------------------- loss
def _weight_bias(value, y_train: np.ndarray) -> float:
    """Peso da classe 'mau' (mesma regra dos outros algoritmos); sem peso = 1."""
    w = resolve_weight_bias(value, y_train)
    return 1.0 if w is None else w


class LossWrapper:
    """Interface única: loss(logits, y_float) e proba(logits) -> P(mau).

    n_outputs = 1 para bce/mse (um logit) e 2 para cross_entropy (dois logits).
    """

    def __init__(self, name: str, weight_bias: float, device):
        self.name = name.lower()
        self.weight_bias = weight_bias
        self.pw = torch.tensor(weight_bias, device=device)
        if self.name == "cross_entropy":
            self.n_outputs = 2
            self.ce = nn.CrossEntropyLoss(weight=torch.tensor([1.0, weight_bias], device=device))
        elif self.name in ("bce", "mse"):
            self.n_outputs = 1
        else:
            raise ValueError(f"loss '{name}' inválida (bce | cross_entropy | mse)")

    def __call__(self, logits, y):
        if self.name == "cross_entropy":
            return self.ce(logits, y.long())
        z = logits.squeeze(-1)
        if self.name == "bce":
            return F.binary_cross_entropy_with_logits(z, y, pos_weight=self.pw)
        # mse sobre a probabilidade, ponderado pela classe
        weights = torch.where(y > 0.5, self.pw, torch.ones_like(y))
        return ((torch.sigmoid(z) - y) ** 2 * weights).mean()

    def proba(self, logits):
        if self.name == "cross_entropy":
            return torch.softmax(logits, dim=1)[:, 1]
        return torch.sigmoid(logits.squeeze(-1))


def build_loss(cfg: dict, y_train: np.ndarray, device) -> LossWrapper:
    return LossWrapper(cfg["name"], _weight_bias(cfg.get("weight_bias"), y_train), device)
