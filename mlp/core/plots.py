"""Gráfico específico da MLP: curvas de aprendizado."""
import matplotlib.pyplot as plt

from common.plot_style import C_BAD, C_GOOD, C_INK, C_MUTED, apply_style, save

apply_style()


def plot_learning_curves(history, best, monitor, path, title):
    steps = history["step"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    ax1.plot(steps, history["train_loss"], color=C_GOOD, lw=1.5, label="treino")
    ax1.plot(steps, history["val_loss"], color=C_BAD, lw=1.5, label="validação")
    ax1.set_xlabel("step")
    ax1.set_ylabel("loss")
    ax1.set_title("Loss")
    ax1.legend(frameon=False)

    ax2.plot(steps, history[monitor], color=C_BAD, lw=1.5)
    ax2.scatter([best["step"]], [best["score"]], s=60, color=C_INK, zorder=3,
                label=f"melhor checkpoint (step {best['step']})")
    ax2.set_xlabel("step")
    ax2.set_ylabel(monitor)
    ax2.set_title(f"Métrica monitorada na validação: {monitor}")
    ax2.legend(frameon=False)
    for ax in (ax1, ax2):
        ax.axvline(best["step"], color=C_MUTED, ls="--", lw=1)
    fig.suptitle(title, y=1.02)
    save(fig, path)
