"""Divisão aleatória das linhas em treino / validação / teste."""
import hashlib

import numpy as np
from sklearn.model_selection import train_test_split


def make_split(y: np.ndarray, split_cfg: dict):
    """Sorteia as linhas em treino / validação / teste.

    Com stratify=true o sorteio é feito dentro de cada classe, mantendo a proporção
    de bons/maus nos três conjuntos. A mesma seed gera sempre a mesma divisão.
    """
    fr = {k: float(split_cfg[k]) for k in ("train", "val", "test")}
    if any(v <= 0 for v in fr.values()) or abs(sum(fr.values()) - 1) > 1e-6:
        raise ValueError(f"split.train/val/test devem ser > 0 e somar 1 (recebido {fr})")
    seed, stratify = split_cfg["seed"], split_cfg.get("stratify", True)
    idx = np.arange(len(y))
    rest, test = train_test_split(idx, test_size=fr["test"], shuffle=True, random_state=seed,
                                  stratify=y if stratify else None)
    train, val = train_test_split(rest, test_size=fr["val"] / (fr["train"] + fr["val"]),
                                  shuffle=True, random_state=seed,
                                  stratify=y[rest] if stratify else None)
    # garantia contra vazamento: nenhuma linha em mais de um conjunto e nenhuma esquecida
    assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))
    assert len(train) + len(val) + len(test) == len(y)
    return np.sort(train), np.sort(val), np.sort(test)


def split_id(train, val, test) -> str:
    """Identificador curto da divisão: execuções com o mesmo id usaram exatamente as mesmas linhas
    em treino, validação e teste (usado para garantir comparações justas)."""
    key = "|".join(",".join(map(str, s)) for s in (train, val, test))
    return hashlib.md5(key.encode()).hexdigest()[:10]
