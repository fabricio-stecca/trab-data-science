"""Pré-processamento: one-hot nos categóricos, log + escala nos numéricos.

O ColumnTransformer é sempre ajustado (fit) apenas na parte de treino de cada
fold, para não vazar informação do teste.
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, MinMaxScaler, OneHotEncoder,
                                   RobustScaler, StandardScaler)

from common.dataset import CATEGORICAL, NUMERIC

SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


def _scaler(name):
    if name in (None, "none"):
        return "passthrough"
    if name not in SCALERS:
        raise ValueError(f"scaler '{name}' inválido. Opções: {list(SCALERS) + ['none']}")
    return SCALERS[name]()


def build_preprocessor(data_cfg: dict, features: list[str]) -> ColumnTransformer:
    numeric = [c for c in NUMERIC if c in features]
    categorical = [c for c in CATEGORICAL if c in features]
    log_cols = [c for c in (data_cfg.get("log_transform") or []) if c in numeric]
    plain_cols = [c for c in numeric if c not in log_cols]

    min_freq = data_cfg.get("rare_category_min_frequency")
    onehot = OneHotEncoder(
        sparse_output=False,
        min_frequency=min_freq,
        handle_unknown="infrequent_if_exist" if min_freq else "ignore",
    )

    transformers = []
    if log_cols:
        transformers.append(("num_log", Pipeline([
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", _scaler(data_cfg.get("scaler"))),
        ]), log_cols))
    if plain_cols:
        transformers.append(("num", _scaler(data_cfg.get("scaler")), plain_cols))
    if categorical:
        transformers.append(("cat", onehot, categorical))
    return ColumnTransformer(transformers, verbose_feature_names_out=False)
