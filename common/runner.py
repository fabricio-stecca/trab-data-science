"""Protocolo de avaliação comum a todos os algoritmos.

Cada algoritmo só fornece uma função `fit_fn` que treina o modelo e devolve uma função
`predict_proba(X) -> P(mau)`. Todo o resto é igual para todos:

    1. divisão aleatória treino / validação / teste (common/protocolo.yaml)
    2. pré-processamento ajustado SÓ no treino
    3. treino (fit_fn) — pode usar a validação (ex.: checkpoint da MLP), nunca o teste
    4. limiar de decisão escolhido na validação
    5. avaliação única no teste
    6. artefatos em resultados/runs/<algoritmo>/<nome>_<data-hora>/ e uma linha em resultados/resultados.csv
"""
import argparse
import json
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import yaml

from common import PROJECT_ROOT
from common.config import load_configs
from common.dataset import FEATURES, TARGET, load_raw
from common.metrics import compute_metrics, find_threshold
from common.plots import plot_confusion, plot_roc
from common.preprocessing import build_preprocessor
from common.reporting import append_results, print_comparison
from common.splits import make_split, split_id


def run_experiment(cfg: dict, algorithm: str, fit_fn) -> dict:
    """fit_fn(cfg, X_tr, y_tr, X_va, y_va, run_dir) -> (predict_proba, params: dict)"""
    exp, ev, dcfg = cfg["experiment"], cfg["evaluation"], cfg["data"]

    df = load_raw(PROJECT_ROOT / dcfg["path"])
    features = [f for f in FEATURES if f not in (dcfg.get("drop_features") or [])]
    X_df, y = df[features], df[TARGET].to_numpy()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = PROJECT_ROOT / exp["output_dir"] / algorithm / f"{exp['name']}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    # 1. divisão aleatória
    tr, va, te = make_split(y, cfg["split"])
    sid = split_id(tr, va, te)
    pd.DataFrame({"idx": np.concatenate([tr, va, te]),
                  "conjunto": ["treino"] * len(tr) + ["validacao"] * len(va) + ["teste"] * len(te)}
                 ).sort_values("idx").to_csv(run_dir / "split.csv", index=False)

    print(f"\n=== [{algorithm}] {exp['name']} ===")
    print(f"  divisão {sid}: treino {len(tr)} | validação {len(va)} | teste {len(te)} "
          f"({y[tr].mean():.0%}/{y[va].mean():.0%}/{y[te].mean():.0%} maus)")

    # 2. pré-processamento ajustado só no treino
    prep = build_preprocessor(dcfg, features)
    X_tr = prep.fit_transform(X_df.iloc[tr])
    X_va = prep.transform(X_df.iloc[va])
    X_te = prep.transform(X_df.iloc[te])
    joblib.dump(prep, run_dir / "preprocessador.joblib")

    # 3. treino (o teste não é passado para fit_fn)
    t0 = time.time()
    predict_proba, params = fit_fn(cfg, X_tr, y[tr], X_va, y[va], run_dir)
    elapsed = time.time() - t0

    # 4. limiar escolhido só na validação
    p_va = predict_proba(X_va)
    thr_cfg = ev["threshold"]
    thr = (find_threshold(y[va], p_va, thr_cfg, ev["cost_fn"], ev["cost_fp"])
           if isinstance(thr_cfg, str) else float(thr_cfg))
    val_m = compute_metrics(y[va], p_va, thr, ev["cost_fn"], ev["cost_fp"])

    # 5. avaliação única no teste
    p_te = predict_proba(X_te)
    test_m = compute_metrics(y[te], p_te, thr, ev["cost_fn"], ev["cost_fp"])
    pd.DataFrame({"idx": te, "y_true": y[te], "proba_mau": p_te,
                  "y_pred": (p_te >= thr).astype(int)}).to_csv(run_dir / "predicoes_teste.csv", index=False)

    # 6. artefatos
    result = {
        "algorithm": algorithm, "name": exp["name"], "params": params,
        "run_dir": str(run_dir.relative_to(PROJECT_ROOT)) if run_dir.is_relative_to(PROJECT_ROOT) else str(run_dir),
        "seconds": round(elapsed, 2), "n_inputs": int(X_tr.shape[1]), "threshold": thr,
        "split_id": sid, "split_sizes": {"treino": len(tr), "validacao": len(va), "teste": len(te)},
        "validacao": val_m, "teste": test_m,
    }
    with open(run_dir / "metricas.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=float)
    plot_confusion(test_m, run_dir / "matriz_confusao.png", f"{algorithm} — {exp['name']}")
    plot_roc(y[te], p_te, test_m["auc"], run_dir / "curva_roc.png", f"{algorithm} — {exp['name']}")
    append_results(cfg, result)

    print(f"  limiar={thr:.2f}  ({elapsed:.1f}s)")
    print(f"  validação: acc={val_m['accuracy']:.3f}  recall={val_m['recall']:.3f}  f1={val_m['f1']:.3f}  "
          f"auc={val_m['auc']:.3f}  custo_médio={val_m['cost_mean']:.3f}")
    print(f"  teste:     acc={test_m['accuracy']:.3f}  recall={test_m['recall']:.3f}  f1={test_m['f1']:.3f}  "
          f"auc={test_m['auc']:.3f}  custo_médio={test_m['cost_mean']:.3f}")
    return result


def main(algorithm: str, default_config, fit_fn, description: str = ""):
    """Linha de comando comum: -c <yamls> --set chave=valor."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--config", nargs="*", help="um ou mais YAMLs (aceita curingas)")
    parser.add_argument("--set", nargs="*", default=[], metavar="CHAVE=VALOR",
                        help="sobrescreve valores, ex.: model.n_neighbors=7")
    args = parser.parse_args()

    configs = load_configs(args.config, args.set, default_config)
    print(f"[{algorithm}] {len(configs)} configuração(ões) para rodar")
    results = [run_experiment(cfg, algorithm, fit_fn) for cfg in configs]
    print_comparison(results, configs, configs[0]["evaluation"]["main_metric"])
    return results


def fit_sklearn(build_model):
    """Cria uma fit_fn para modelos do scikit-learn: build_model(cfg, y_train) -> estimador."""
    def fit_fn(cfg, X_tr, y_tr, X_va, y_va, run_dir):
        model = build_model(cfg, y_tr)
        model.fit(X_tr, y_tr)          # a validação não é usada no ajuste destes modelos
        joblib.dump(model, run_dir / "modelo.joblib")
        params = {"modelo": cfg["model"],
                  "pre_processamento": {k: cfg["data"][k] for k in ("log_transform", "scaler")}}
        return (lambda X: model.predict_proba(X)[:, 1]), params
    return fit_fn


def resolve_weight_bias(value, y_train) -> float | None:
    """Peso da classe 'mau' calculado SÓ com o treino. auto = n_bom/n_mau; null = sem peso."""
    if value is None:
        return None
    if value == "auto":
        n_pos = int(np.sum(y_train))
        return (len(y_train) - n_pos) / n_pos
    return float(value)
