"""Gráficos de avaliação usados por todos os algoritmos: matriz de confusão e curva ROC."""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

from common.plot_style import C_GOOD, C_INK, C_MUTED, apply_style, save

apply_style()


def draw_confusion(ax, m: dict, title: str):
    """Desenha a matriz de confusão (dict com tn, fp, fn, tp) em um eixo."""
    mat = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    total = mat.sum()
    ax.imshow(mat, cmap="Blues", vmin=0, vmax=mat.max() * 1.15)
    names = [["VN", "FP"], ["FN", "VP"]]
    for i in range(2):
        for j in range(2):
            v = mat[i, j]
            ax.text(j, i, f"{names[i][j]}\n{v}\n({v / total:.0%})", ha="center", va="center",
                    fontsize=10, color="white" if v > mat.max() * 0.6 else C_INK)
    ax.set_xticks([0, 1], ["bom", "mau"])
    ax.set_yticks([0, 1], ["bom", "mau"])
    ax.set_xlabel("classe prevista")
    ax.set_ylabel("classe real")
    ax.grid(False)
    ax.set_title(title, fontsize=9)


def plot_confusion(m: dict, path, title: str):
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    draw_confusion(ax, m, f"Matriz de confusão — teste (limiar {m['threshold']:.2f})\n{title}")
    save(fig, path)


def plot_roc(y_true, proba, auc: float, path, title: str):
    fpr, tpr, _ = roc_curve(y_true, proba)
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.plot(fpr, tpr, color=C_GOOD, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], ls="--", color=C_MUTED, lw=1, label="aleatório")
    ax.set_xlabel("taxa de falsos positivos (1 − especificidade)")
    ax.set_ylabel("taxa de verdadeiros positivos (recall)")
    ax.set_title(f"Curva ROC — teste\n{title}", fontsize=9)
    ax.legend(frameon=False, loc="lower right")
    save(fig, path)
