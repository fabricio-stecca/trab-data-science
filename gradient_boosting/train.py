"""Gradient Boosting (HistGradientBoostingClassifier) — scikit-learn.

Uso (a partir da raiz do projeto):
    python gradient_boosting/train.py
    python gradient_boosting/train.py -c gradient_boosting/configs/experiments/grade.yaml
    python gradient_boosting/train.py --set model.learning_rate=0.05

Treina árvores em sequência: cada nova árvore é ajustada para corrigir os erros (gradiente
da loss) do conjunto das anteriores, com passo `learning_rate`. P(mau) = sigmoide da soma.
A divisão, o limiar e as métricas seguem o protocolo comum (common/protocolo.yaml).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # permite importar common/

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

from common import PROJECT_ROOT  # noqa: E402
from common.runner import fit_sklearn, main, resolve_weight_bias  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "gradient_boosting" / "configs" / "default.yaml"


def build_model(cfg, y_train):
    m = cfg["model"]
    w = resolve_weight_bias(m["weight_bias"], y_train)   # calculado só com o treino
    return HistGradientBoostingClassifier(
        learning_rate=m["learning_rate"],
        max_iter=m["max_iter"],
        max_depth=m["max_depth"],
        max_leaf_nodes=m["max_leaf_nodes"],
        min_samples_leaf=m["min_samples_leaf"],
        l2_regularization=m["l2_regularization"],
        class_weight=None if w is None else {0: 1.0, 1: w},
        early_stopping=False,   # nº de árvores fixo (max_iter); a escolha é feita na validação
        random_state=cfg["experiment"]["seed"],
    )


if __name__ == "__main__":
    main("gradient_boosting", DEFAULT_CONFIG, fit_sklearn(build_model), __doc__)
