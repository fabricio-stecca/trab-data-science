"""Cores e estilo dos gráficos (iguais na EDA e no modelo)."""
import matplotlib

matplotlib.use("Agg")  # gera arquivos PNG sem abrir janelas
import matplotlib.pyplot as plt  # noqa: E402

C_GOOD = "#2a78d6"   # bom pagador (azul)
C_BAD = "#eb6834"    # mau pagador (laranja)
C_INK = "#0b0b0b"    # texto principal
C_MUTED = "#52514e"  # texto secundário / linhas de referência
C_GRID = "#e4e3df"   # grade e eixos
CLASS_COLORS = {0: C_GOOD, 1: C_BAD}


def apply_style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.edgecolor": C_GRID, "axes.labelcolor": C_MUTED,
        "xtick.color": C_MUTED, "ytick.color": C_MUTED,
        "axes.grid": True, "grid.color": C_GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
