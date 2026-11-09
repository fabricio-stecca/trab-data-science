"""Treino da MLP: avaliação periódica na validação, checkpoint do melhor modelo
e parada antecipada.

Só recebe dados de treino e de validação — o teste nunca passa por aqui.
"""
import json
import random
from pathlib import Path

import numpy as np
import torch

from common.metrics import compute_metrics, find_threshold
from mlp.core.factory import build_loss, build_optimizer, build_scheduler
from mlp.core.model import build_model

# monitor -> True se menor é melhor
MONITORS = {"val_loss": True, "val_cost": True, "val_auc": False, "val_f1": False}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(name: str) -> torch.device:
    name = (name or "auto").lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        print("[aviso] CUDA indisponível — usando CPU.")
        return torch.device("cpu")
    return torch.device(name)


@torch.no_grad()
def predict_proba(model, loss_fn, X: torch.Tensor) -> np.ndarray:
    model.eval()
    return loss_fn.proba(model(X)).cpu().numpy()


def _to_plain(obj):
    """Converte tipos numpy para tipos nativos (o checkpoint é lido com weights_only=True)."""
    return json.loads(json.dumps(obj, default=float))


def load_checkpoint(path, device):
    return torch.load(path, map_location=device, weights_only=True)


def train_model(cfg: dict, X_tr: np.ndarray, y_tr: np.ndarray, X_val: np.ndarray,
                y_val: np.ndarray, device: torch.device, ckpt_path: Path):
    """Treina, salvando em `ckpt_path` sempre que o monitor de validação melhora.

    Retorna (modelo com os pesos do melhor checkpoint, loss_fn, histórico, info do melhor).
    """
    set_seed(cfg["experiment"]["seed"])
    tcfg, ev = cfg["training"], cfg["evaluation"]
    monitor = tcfg["monitor"]
    if monitor not in MONITORS:
        raise ValueError(f"training.monitor '{monitor}' inválido. Opções: {list(MONITORS)}")
    minimize = MONITORS[monitor]
    es = tcfg["early_stopping"]
    min_delta = float(es.get("min_delta", 0.0))
    eval_every = int(tcfg["eval_every_steps"])

    Xt = torch.tensor(X_tr, dtype=torch.float32, device=device)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=device)
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.float32, device=device)

    loss_fn = build_loss(tcfg["loss"], y_tr, device)          # weight_bias vem só do treino
    model = build_model(cfg["model"], Xt.shape[1], loss_fn.n_outputs).to(device)
    optimizer = build_optimizer(model.parameters(), tcfg["optimizer"])
    scheduler = build_scheduler(optimizer, tcfg.get("scheduler"), tcfg["epochs"])

    history = {k: [] for k in ("step", "epoch", "train_loss", "val_loss", "val_auc",
                               "val_f1", "val_cost", "threshold", "lr")}
    best = {"score": np.inf if minimize else -np.inf, "step": 0, "epoch": 0}
    state = {"step": 0, "evals_without_improvement": 0, "run_loss": 0.0, "run_n": 0}

    def evaluate(epoch: int) -> bool:
        """Avalia na validação, salva checkpoint se melhorou. Retorna True para parar."""
        model.eval()
        with torch.no_grad():
            logits = model(Xv)
            vloss = float(loss_fn(logits, yv).item())
            vproba = loss_fn.proba(logits).cpu().numpy()
        thr_cfg = ev["threshold"]
        thr = (find_threshold(y_val, vproba, thr_cfg, ev["cost_fn"], ev["cost_fp"])
               if isinstance(thr_cfg, str) else float(thr_cfg))
        m = compute_metrics(y_val, vproba, thr, ev["cost_fn"], ev["cost_fp"])
        scores = {"val_loss": vloss, "val_auc": m["auc"], "val_f1": m["f1"], "val_cost": m["cost_mean"]}

        history["step"].append(state["step"])
        history["epoch"].append(epoch)
        history["train_loss"].append(state["run_loss"] / max(state["run_n"], 1))
        for k in ("val_loss", "val_auc", "val_f1", "val_cost"):
            history[k].append(scores[k])
        history["threshold"].append(thr)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        state["run_loss"], state["run_n"] = 0.0, 0

        score = scores[monitor]
        improved = (score < best["score"] - min_delta) if minimize else (score > best["score"] + min_delta)
        if improved:
            best.update(score=score, step=state["step"], epoch=epoch, threshold=thr,
                        val_metrics={**m, "loss": vloss})
            torch.save(_to_plain({
                "monitor": monitor, "score": score, "step": state["step"], "epoch": epoch,
                "threshold": thr, "val_metrics": best["val_metrics"],
                "in_features": int(Xt.shape[1]), "n_outputs": loss_fn.n_outputs,
                "config": cfg,
            }) | {"model_state_dict": model.state_dict()}, ckpt_path)
            state["evals_without_improvement"] = 0
        else:
            state["evals_without_improvement"] += 1
        model.train()
        return es.get("enabled", True) and state["evals_without_improvement"] >= es["patience"]

    gen = torch.Generator(device="cpu").manual_seed(cfg["experiment"]["seed"])
    n, bs = len(Xt), tcfg["batch_size"]
    stop, epoch = False, 0
    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        perm = torch.randperm(n, generator=gen).to(device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2 and cfg["model"].get("batch_norm"):
                continue  # BatchNorm não aceita batch de tamanho 1
            optimizer.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            if tcfg.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            optimizer.step()
            state["step"] += 1
            state["run_loss"] += loss.item() * len(idx)
            state["run_n"] += len(idx)
            if state["step"] % eval_every == 0 and evaluate(epoch):
                stop = True
                break
        if stop:
            break
        if scheduler is not None:
            scheduler.step()

    if not stop and state["step"] % eval_every != 0:
        evaluate(epoch)  # avaliação final se o último step não coincidiu com uma avaliação

    best["stopped_early"] = stop
    best["total_steps"] = state["step"]
    best["total_epochs"] = epoch
    ckpt = load_checkpoint(ckpt_path, device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, loss_fn, history, best
