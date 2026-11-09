"""MLP (rede neural multicamadas) em PyTorch.

Uso (a partir da raiz do projeto):
    python mlp/train.py                                          # mlp/configs/default.yaml
    python mlp/train.py -c mlp/configs/experiments/tanh.yaml     # um experimento
    python mlp/train.py -c "mlp/configs/experiments/*.yaml"      # vários experimentos
    python mlp/train.py -c mlp/configs/sweep_exemplo.yaml        # grade de combinações
    python mlp/train.py --set model.activation=gelu device=cuda  # sobrescreve valores

A divisão, o limiar e as métricas seguem o protocolo comum (common/protocolo.yaml).
Específico da MLP: a cada `eval_every_steps` o modelo é avaliado na validação, o melhor
checkpoint é salvo em best_model.pt e o treino para se não houver melhora.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # permite importar common/ e mlp/

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from common import PROJECT_ROOT  # noqa: E402
from common.runner import main  # noqa: E402
from mlp.core.plots import plot_learning_curves  # noqa: E402
from mlp.core.trainer import predict_proba, resolve_device, train_model  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "mlp" / "configs" / "default.yaml"


def fit_mlp(cfg, X_tr, y_tr, X_va, y_va, run_dir):
    device = resolve_device(cfg["device"])
    model, loss_fn, history, best = train_model(cfg, X_tr, y_tr, X_va, y_va, device, run_dir / "best_model.pt")
    pd.DataFrame(history).to_csv(run_dir / "historico_treino.csv", index=False)
    monitor = cfg["training"]["monitor"]
    plot_learning_curves(history, best, monitor, run_dir / "curvas_aprendizado.png", cfg["experiment"]["name"])

    parada = "parada antecipada" if best["stopped_early"] else "limite de épocas"
    print(f"  device={device} | melhor checkpoint: step {best['step']} (época {best['epoch']}), "
          f"{monitor}={best['score']:.4f} [{best['total_steps']} steps, {parada}]")

    t = cfg["training"]
    params = {
        "camadas": cfg["model"]["hidden_layers"], "ativacao": cfg["model"]["activation"],
        "dropout": cfg["model"]["dropout"], "otimizador": t["optimizer"]["name"],
        "lr": t["optimizer"]["lr"], "weight_decay": t["optimizer"]["weight_decay"],
        "loss": t["loss"]["name"], "weight_bias": t["loss"]["weight_bias"],
        "batch": t["batch_size"], "monitor": monitor, "melhor_step": best["step"],
    }

    def predict(X):
        return predict_proba(model, loss_fn, torch.tensor(X, dtype=torch.float32, device=device))
    return predict, params


if __name__ == "__main__":
    main("mlp", DEFAULT_CONFIG, fit_mlp, __doc__)
