"""Random Forest — scikit-learn.

Uso (a partir da raiz do projeto):
    python random_forest/train.py
    python random_forest/train.py -c random_forest/configs/experiments/grade.yaml
    python random_forest/train.py --set model.max_depth=6

Treina muitas árvores de decisão, cada uma numa amostra bootstrap do treino e escolhendo
entre um subconjunto aleatório de atributos a cada divisão. P(mau) = média das árvores.
A divisão, o limiar e as métricas seguem o protocolo comum (common/protocolo.yaml).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # permite importar common/

from sklearn.ensemble import RandomForestClassifier  # noqa: E402

from common import PROJECT_ROOT  # noqa: E402
from common.runner import fit_sklearn, main, resolve_weight_bias  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "random_forest" / "configs" / "default.yaml"


def build_model(cfg, y_train):
    m = cfg["model"]
    w = resolve_weight_bias(m["weight_bias"], y_train)   # calculado só com o treino
    return RandomForestClassifier(
        n_estimators=m["n_estimators"],
        max_depth=m["max_depth"],
        min_samples_leaf=m["min_samples_leaf"],
        max_features=m["max_features"],
        class_weight=None if w is None else {0: 1.0, 1: w},
        random_state=cfg["experiment"]["seed"],
        n_jobs=-1,
    )


if __name__ == "__main__":
    main("random_forest", DEFAULT_CONFIG, fit_sklearn(build_model), __doc__)
