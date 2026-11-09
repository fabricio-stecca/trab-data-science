"""Métricas de avaliação (classe positiva = 1 = mau pagador).

Com TP, TN, FP, FN da matriz de confusão:
    acurácia            = (TP + TN) / (TP + TN + FP + FN)
    precisão            = TP / (TP + FP)
    recall (sensib.)    = TP / (TP + FN)
    especificidade      = TN / (TN + FP)
    F1                  = 2 · precisão · recall / (precisão + recall)
    acurácia balanceada = (recall + especificidade) / 2
    custo total         = cost_fn · FN + cost_fp · FP      (UCI: 5·FN + 1·FP)
    custo médio         = custo total / n
    AUC-ROC             = área sob a curva ROC (independe do limiar)
"""
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

METRIC_NAMES = ["accuracy", "balanced_accuracy", "precision", "recall", "specificity",
                "f1", "auc", "cost_total", "cost_mean"]
# métricas onde menor é melhor
LOWER_IS_BETTER = {"cost_total", "cost_mean"}
THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)


def _safe_div(a, b) -> float:
    return float(a) / float(b) if b else 0.0


def compute_metrics(y_true, proba, threshold=0.5, cost_fn=5.0, cost_fp=1.0) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    cost = float(cost_fn * fn + cost_fp * fp)
    try:
        auc = float(roc_auc_score(y_true, proba))
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": _safe_div(tp + tn, len(y_true)),
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "auc": auc,
        "cost_total": cost,
        "cost_mean": cost / len(y_true),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "threshold": float(threshold),
    }


def find_threshold(y_true, proba, criterion: str, cost_fn=5.0, cost_fp=1.0) -> float:
    """Limiar da grade 0.05..0.95 que minimiza o custo (optimize_cost) ou maximiza o F1
    (optimize_f1). Deve receber apenas dados de VALIDAÇÃO."""
    if criterion not in ("optimize_cost", "optimize_f1"):
        raise ValueError("evaluation.threshold deve ser um número, 'optimize_cost' ou 'optimize_f1'")
    y = np.asarray(y_true).astype(bool)
    pred = np.asarray(proba)[None, :] >= THRESHOLD_GRID[:, None]   # (n_limiares, n_amostras)
    tp = (pred & y).sum(1)
    fp = (pred & ~y).sum(1)
    fn = (~pred & y).sum(1)
    if criterion == "optimize_cost":
        score = -(cost_fn * fn + cost_fp * fp).astype(float)
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            score = np.nan_to_num(2 * tp / (2 * tp + fp + fn))
    best = np.flatnonzero(score == score.max())
    return float(THRESHOLD_GRID[best[len(best) // 2]])  # em empates, o limiar do meio da faixa
