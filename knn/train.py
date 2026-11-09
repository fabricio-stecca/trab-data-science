"""KNN (k vizinhos mais próximos) — scikit-learn.

Uso (a partir da raiz do projeto):
    python knn/train.py                                     # knn/configs/default.yaml
    python knn/train.py -c knn/configs/experiments/grade.yaml   # várias parametrizações
    python knn/train.py --set model.n_neighbors=7

Classifica cada cliente pela classe dos k clientes de treino mais parecidos (menor distância).
P(mau) = proporção (ponderada, se weights=distance) de maus entre os k vizinhos.
A divisão, o limiar e as métricas seguem o protocolo comum (common/protocolo.yaml).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # permite importar common/

from sklearn.neighbors import KNeighborsClassifier  # noqa: E402

from common import PROJECT_ROOT  # noqa: E402
from common.runner import fit_sklearn, main  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "knn" / "configs" / "default.yaml"


def build_model(cfg, y_train):
    m = cfg["model"]
    # O KNN não aceita peso de classe; o desbalanceamento é tratado pelo limiar (validação).
    return KNeighborsClassifier(n_neighbors=m["n_neighbors"], weights=m["weights"], p=m["p"])


if __name__ == "__main__":
    main("knn", DEFAULT_CONFIG, fit_sklearn(build_model), __doc__)
