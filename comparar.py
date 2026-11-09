"""Comparação final entre os algoritmos (MLP, KNN, Random Forest, Gradient Boosting).

Uso (a partir da raiz do projeto, depois de treinar os algoritmos):
    python comparar.py

Lê resultados/resultados.csv e, para cada algoritmo:
    1. considera a execução mais recente de cada configuração;
    2. usa só execuções feitas na MESMA divisão treino/validação/teste;
    3. escolhe a melhor configuração pela VALIDAÇÃO (evaluation.main_metric do protocolo);
    4. relata o desempenho dessa configuração no TESTE.

Saídas em resultados/comparacao/:
    tabela_comparativa.csv / .md   melhor configuração de cada algoritmo (métricas de teste)
    todas_configuracoes.csv        todas as parametrizações testadas (validação e teste)
    matrizes_confusao.png          matriz de confusão (teste) de cada algoritmo
    curvas_roc.png                 curvas ROC (teste) sobrepostas
    metricas_teste.png             recall, F1, AUC e custo médio lado a lado
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from common import PROJECT_ROOT  # noqa: E402
from common.config import PROTOCOL_CONFIG  # noqa: E402
from common.metrics import LOWER_IS_BETTER  # noqa: E402
from common.plot_style import C_INK, C_MUTED, apply_style, save  # noqa: E402
from common.plots import draw_confusion  # noqa: E402

apply_style()
ORDER = ["mlp", "knn", "random_forest", "gradient_boosting"]
NAMES = {"mlp": "MLP", "knn": "KNN", "random_forest": "Random Forest", "gradient_boosting": "Gradient Boosting"}
# cores categóricas em ordem fixa (cada algoritmo sempre com a mesma cor)
COLORS = {"mlp": "#2a78d6", "knn": "#eb6834", "random_forest": "#1baf7a", "gradient_boosting": "#eda100"}
TEST_COLS = {"teste_accuracy": "acurácia", "teste_balanced_accuracy": "acurácia_bal.",
             "teste_precision": "precisão", "teste_recall": "recall", "teste_specificity": "especificidade",
             "teste_f1": "f1", "teste_auc": "auc", "teste_cost_mean": "custo_médio"}


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


def main():
    protocol = yaml.safe_load(PROTOCOL_CONFIG.read_text(encoding="utf-8"))
    metric = protocol["evaluation"]["main_metric"]
    results_file = PROJECT_ROOT / protocol["experiment"]["results_file"]
    out = PROJECT_ROOT / "resultados" / "comparacao"
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_file)
    df = df.sort_values("data_hora").drop_duplicates(["algoritmo", "experimento"], keep="last")

    # só execuções na mesma divisão (a mais usada); as demais são descartadas com aviso
    main_split = df["split_id"].value_counts().idxmax()
    other = df[df["split_id"] != main_split]
    if len(other):
        print(f"[aviso] {len(other)} execução(ões) com outra divisão foram ignoradas: "
              f"{sorted(other['experimento'].unique())}")
    df = df[df["split_id"] == main_split]

    algos = [a for a in ORDER if a in set(df["algoritmo"])] + sorted(set(df["algoritmo"]) - set(ORDER))
    asc = metric in LOWER_IS_BETTER
    best_rows, n_configs = [], {}
    for a in algos:
        d = df[df["algoritmo"] == a].sort_values(f"val_{metric}", ascending=asc)
        best_rows.append(d.iloc[0])
        n_configs[a] = len(d)
        if len(d) < 2:
            print(f"[aviso] {NAMES.get(a, a)}: só 1 configuração testada (o enunciado pede pelo menos 2).")
    best = pd.DataFrame(best_rows)

    # ---------------------------------------------------------------- tabelas
    table = pd.DataFrame({
        "algoritmo": [NAMES.get(a, a) for a in best["algoritmo"]],
        "configuração escolhida": best["experimento"].values,
        "nº configs": [n_configs[a] for a in best["algoritmo"]],
        f"val_{metric}": best[f"val_{metric}"].values,
        "limiar": best["limiar"].values,
        **{label: best[col].values for col, label in TEST_COLS.items()},
    })
    table.to_csv(out / "tabela_comparativa.csv", index=False)
    (out / "tabela_comparativa.md").write_text(
        f"Divisão `{main_split}` — melhor configuração de cada algoritmo escolhida pela validação "
        f"(`{metric}`); métricas abaixo são do **teste**.\n\n" + md_table(table) + "\n", encoding="utf-8")

    all_cols = ["algoritmo", "experimento", "limiar", f"val_{metric}", "val_auc"] + list(TEST_COLS)
    df.assign(_o=df["algoritmo"].map({a: i for i, a in enumerate(algos)})) \
      .sort_values(["_o", f"val_{metric}"], ascending=[True, asc])[all_cols] \
      .to_csv(out / "todas_configuracoes.csv", index=False)

    # ---------------------------------------------------------------- figuras
    runs = {a: PROJECT_ROOT / p for a, p in zip(best["algoritmo"], best["pasta"])}

    fig, axes = plt.subplots(1, len(algos), figsize=(4 * len(algos), 3.8))
    for ax, a in zip(np.atleast_1d(axes), algos):
        m = json.loads((runs[a] / "metricas.json").read_text(encoding="utf-8"))["teste"]
        draw_confusion(ax, m, f"{NAMES.get(a, a)}\n(limiar {m['threshold']:.2f}, custo médio {m['cost_mean']:.3f})")
        if ax is not np.atleast_1d(axes)[0]:
            ax.set_ylabel("")
    fig.suptitle("Matrizes de confusão no teste — melhor configuração de cada algoritmo", y=1.04)
    save(fig, out / "matrizes_confusao.png")

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for a in algos:
        p = pd.read_csv(runs[a] / "predicoes_teste.csv")
        fpr, tpr, _ = roc_curve(p["y_true"], p["proba_mau"])
        auc = best.loc[best["algoritmo"] == a, "teste_auc"].iloc[0]
        ax.plot(fpr, tpr, lw=2, color=COLORS.get(a, C_INK), label=f"{NAMES.get(a, a)} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=C_MUTED, label="aleatório")
    ax.set_xlabel("taxa de falsos positivos (1 − especificidade)")
    ax.set_ylabel("taxa de verdadeiros positivos (recall)")
    ax.set_title("Curvas ROC no teste")
    ax.legend(frameon=False, loc="lower right")
    save(fig, out / "curvas_roc.png")

    panels = [("teste_recall", "Recall (maus detectados)", False), ("teste_f1", "F1", False),
              ("teste_auc", "AUC-ROC", False), ("teste_cost_mean", "Custo médio (menor é melhor)", True)]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.2), sharey=True)
    y = np.arange(len(algos))
    for ax, (col, title, _) in zip(axes, panels):
        vals = best[col].to_numpy()
        ax.barh(y, vals, color=[COLORS.get(a, C_INK) for a in best["algoritmo"]], height=0.6)
        for yi, v in zip(y, vals):
            ax.text(v + 0.01, yi, f"{v:.3f}", va="center", fontsize=8, color=C_INK)
        ax.set_xlim(0, max(1.0, vals.max() * 1.2))
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [NAMES.get(a, a) for a in best["algoritmo"]])
    axes[0].invert_yaxis()
    fig.suptitle("Métricas no teste — melhor configuração de cada algoritmo (escolhida pela validação)", y=1.06)
    save(fig, out / "metricas_teste.png")

    # ---------------------------------------------------------------- resumo
    print(f"Divisão {main_split} | escolha pela validação: {metric}\n")
    show = table.drop(columns=["acurácia_bal.", "precisão", "especificidade"])
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nSaídas em {out}")


if __name__ == "__main__":
    main()
