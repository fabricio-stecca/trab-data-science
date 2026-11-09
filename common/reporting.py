"""Tabela acumulada de resultados (todos os algoritmos) e comparação entre configurações."""
import json
from datetime import datetime

import pandas as pd

from common import PROJECT_ROOT
from common.metrics import LOWER_IS_BETTER, METRIC_NAMES


def threshold_rule(cfg: dict) -> str:
    t = cfg["evaluation"]["threshold"]
    return t if isinstance(t, str) else f"fixo {float(t):.2f}"


def append_results(cfg: dict, result: dict):
    """Acrescenta uma linha por execução em resultados/resultados.csv."""
    row = {
        "data_hora": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "algoritmo": result["algorithm"],
        "experimento": result["name"],
        "parametros": json.dumps(result["params"], ensure_ascii=False),
        "regra_limiar": threshold_rule(cfg),
        "limiar": round(result["threshold"], 2),
        "split_id": result["split_id"],
        "split_seed": cfg["split"]["seed"],
        "segundos": result["seconds"],
    }
    for k in METRIC_NAMES:
        row[f"val_{k}"] = round(result["validacao"][k], 4)
    for k in METRIC_NAMES:
        row[f"teste_{k}"] = round(result["teste"][k], 4)
    row["pasta"] = result["run_dir"]

    path = PROJECT_ROOT / cfg["experiment"]["results_file"]
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row])
    if path.exists():  # junta pelas colunas, para não desalinhar se o formato mudar
        new = pd.concat([pd.read_csv(path), new], ignore_index=True)
    new.to_csv(path, index=False, encoding="utf-8")


def print_comparison(results: list[dict], configs: list[dict], main_metric: str):
    """Compara as configurações desta execução e escolhe a melhor pela VALIDAÇÃO.
    O teste só é relatado. (A comparação entre algoritmos fica em comparar.py.)"""
    if len(results) < 2:
        return
    if main_metric not in METRIC_NAMES:
        raise ValueError(f"evaluation.main_metric '{main_metric}' inválido. Opções: {METRIC_NAMES}")
    rows = [{
        "experimento": r["name"],
        "limiar": threshold_rule(cfg),
        f"val_{main_metric}": r["validacao"][main_metric],
        "teste_recall": r["teste"]["recall"],
        "teste_f1": r["teste"]["f1"],
        "teste_auc": r["teste"]["auc"],
        "teste_custo_médio": r["teste"]["cost_mean"],
    } for r, cfg in zip(results, configs)]
    df = pd.DataFrame(rows).sort_values(f"val_{main_metric}", ascending=main_metric in LOWER_IS_BETTER)
    print(f"\n=== Comparação das configurações (ordenada pela VALIDAÇÃO: {main_metric}) ===")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    if df["limiar"].nunique() > 1 and main_metric != "auc":
        print("\n[aviso] As configurações usam regras de limiar diferentes. Quando o limiar é otimizado na "
              "validação, as métricas de validação ficam otimistas, então a comparação entre regras "
              "diferentes não é justa. Compare com a mesma regra, ou use evaluation.main_metric=auc.")

    best = df.iloc[0]
    print(f"\nEscolhida pela validação: {best['experimento']}  (val_{main_metric}={best[f'val_{main_metric}']:.3f})")
