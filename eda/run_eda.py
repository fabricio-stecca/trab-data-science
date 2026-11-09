"""Análise exploratória do Statlog (German Credit Data).

Uso (a partir da raiz do projeto):
    python eda/run_eda.py                 # gera tudo em eda/outputs/
    python eda/run_eda.py --out outra/pasta

Saídas:
    eda/outputs/figs/*.png        5 gráficos simples
    eda/outputs/tables/*.csv      tabelas
    eda/outputs/relatorio_eda.md  resumo com os achados principais
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # permite importar common/
import warnings

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.feature_selection import mutual_info_classif  # noqa: E402

from common import PROJECT_ROOT  # noqa: E402
from common.dataset import (  # noqa: E402
    CATEGORICAL, CODEBOOK, DESCRIPTIONS, FEATURES, NUMERIC, TARGET,
    TARGET_LABELS, TYPES, UCI_URL, load_raw,
)
from common.plot_style import (  # noqa: E402
    C_BAD, C_GOOD, C_INK, C_MUTED, CLASS_COLORS, apply_style, save,
)

warnings.filterwarnings("ignore", category=FutureWarning)
apply_style()


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    """DataFrame -> tabela markdown (sem depender do pacote tabulate)."""
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return floatfmt.format(v)
        return str(v)
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [fmt(v) for v in row]) + " |")
    return "\n".join(lines)


def cramers_v(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """Cramér's V (com correção de viés de Bergsma) e p-valor do qui-quadrado."""
    table = pd.crosstab(x, y)
    chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
    n = table.values.sum()
    r, k = table.shape
    phi2 = chi2 / n
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    denom = min(kc - 1, rc - 1)
    return (np.sqrt(phi2c / denom) if denom > 0 else 0.0), p


def ordered_categories(col: str) -> list[str]:
    """Ordem das categorias conforme o dicionário (A40..A410 em ordem correta)."""
    return list(CODEBOOK[col].keys())


# ----------------------------------------------------------------- análises
def section_overview(df, tables):
    rows = []
    for col in FEATURES:
        s = df[col]
        rows.append({
            "atributo": col,
            "descrição": DESCRIPTIONS[col],
            "tipo": TYPES[col],
            "dtype": str(s.dtype),
            "n_valores_distintos": s.nunique(),
            "ausentes": int(s.isna().sum()),
            "exemplos": ", ".join(map(str, s.unique()[:4])),
        })
    info = pd.DataFrame(rows).set_index("atributo")
    info.to_csv(tables / "01_atributos.csv")
    return info


def section_target(df, figs, tables):
    counts = df[TARGET].value_counts().sort_index()
    dist = pd.DataFrame({
        "classe": [TARGET_LABELS[i] for i in counts.index],
        "n": counts.values,
        "%": (counts.values / len(df) * 100).round(1),
    }).set_index("classe")
    dist.to_csv(tables / "02_distribuicao_classes.csv")

    fig, ax = plt.subplots(figsize=(4.2, 3))
    bars = ax.bar(dist.index, dist["n"], color=[C_GOOD, C_BAD], width=0.55)
    for b, (n, pct) in zip(bars, zip(dist["n"], dist["%"])):
        ax.text(b.get_x() + b.get_width() / 2, n + 12, f"{n} ({pct}%)",
                ha="center", color=C_INK)
    ax.set_ylabel("nº de clientes")
    ax.set_ylim(0, dist["n"].max() * 1.15)
    ax.set_title("Distribuição da classe (risco de crédito)")
    ax.grid(axis="x", visible=False)
    save(fig, figs / "01_distribuicao_classes.png")
    return dist


def section_quality(df, tables):
    """Ausentes, duplicatas, inconsistências e categorias raras."""
    findings = {}
    findings["ausentes_total"] = int(df.isna().sum().sum())
    findings["linhas_duplicadas"] = int(df.duplicated().sum())
    # mesmas features com rótulos diferentes (ruído de rótulo)
    dup_feat = df.duplicated(subset=FEATURES, keep=False)
    findings["features_duplicadas_rotulo_conflitante"] = int(
        (df[dup_feat].groupby(FEATURES)[TARGET].nunique() > 1).sum()
    ) if dup_feat.any() else 0

    # categorias documentadas mas ausentes / códigos não documentados
    absent, undocumented, rare = [], [], []
    for col in CATEGORICAL:
        documented = set(CODEBOOK[col])
        observed = set(df[col].unique())
        for c in sorted(documented - observed):
            absent.append(f"{col}={c} ({CODEBOOK[col][c]})")
        for c in sorted(observed - documented):
            undocumented.append(f"{col}={c}")
        freq = df[col].value_counts(normalize=True)
        for c, f in freq.items():
            if f < 0.02:
                rare.append({"atributo": col, "categoria": c,
                             "rótulo": CODEBOOK[col].get(c, c),
                             "n": int((df[col] == c).sum()), "%": round(f * 100, 1)})
    findings["categorias_documentadas_ausentes"] = absent
    findings["codigos_nao_documentados"] = undocumented
    rare_df = pd.DataFrame(rare)
    rare_df.to_csv(tables / "03_categorias_raras.csv", index=False)

    # checagens de domínio nos numéricos
    checks = {
        "duration <= 0": int((df["duration"] <= 0).sum()),
        "credit_amount <= 0": int((df["credit_amount"] <= 0).sum()),
        "age < 18": int((df["age"] < 18).sum()),
        "age > 100": int((df["age"] > 100).sum()),
        "installment_rate fora de 1-4": int((~df["installment_rate"].between(1, 4)).sum()),
        "residence_since fora de 1-4": int((~df["residence_since"].between(1, 4)).sum()),
    }
    # inconsistências semânticas
    checks["job=desempregado(A171) e employment>=1 ano"] = int(
        ((df["job"] == "A171") & df["employment"].isin(["A73", "A74", "A75"])).sum())
    checks["employment=desempregado(A71) e job=qualificado/gerência"] = int(
        ((df["employment"] == "A71") & df["job"].isin(["A173", "A174"])).sum())
    checks["credit_history=sem créditos(A30) e existing_credits>1"] = int(
        ((df["credit_history"] == "A30") & (df["existing_credits"] > 1)).sum())
    findings["checagens"] = checks
    pd.Series(checks, name="ocorrências").to_csv(tables / "04_checagens_consistencia.csv")
    return findings, rare_df


def section_numeric(df, figs, tables):
    desc = df[NUMERIC].describe().T
    desc["assimetria"] = df[NUMERIC].skew()
    desc["curtose"] = df[NUMERIC].kurt()
    q1, q3 = df[NUMERIC].quantile(0.25), df[NUMERIC].quantile(0.75)
    iqr = q3 - q1
    desc["outliers_IQR"] = ((df[NUMERIC] < q1 - 1.5 * iqr) | (df[NUMERIC] > q3 + 1.5 * iqr)).sum()
    desc["média_bom"] = df[df[TARGET] == 0][NUMERIC].mean()
    desc["média_mau"] = df[df[TARGET] == 1][NUMERIC].mean()
    desc.index.name = "atributo"
    desc.round(3).to_csv(tables / "05_estatisticas_numericos.csv")

    # Um único gráfico: boxplot por classe dos 3 numéricos contínuos
    continuous = ["duration", "credit_amount", "age"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, col in zip(axes, continuous):
        data = [df.loc[df[TARGET] == c, col] for c in (0, 1)]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                        medianprops={"color": C_INK, "linewidth": 2},
                        flierprops={"marker": "o", "markersize": 3, "alpha": 0.35,
                                    "markeredgecolor": C_MUTED})
        for patch, cls in zip(bp["boxes"], (0, 1)):
            patch.set_facecolor(CLASS_COLORS[cls])
            patch.set_alpha(0.75)
            patch.set_edgecolor(CLASS_COLORS[cls])
        for i, d in enumerate(data, start=1):
            med = d.median()
            ax.text(i + 0.3, med, f"mediana\n{med:,.0f}".replace(",", "."), va="center",
                    fontsize=7, color=C_INK)
        ax.set_xticks([1, 2], [f"{TARGET_LABELS[0]} pagador", f"{TARGET_LABELS[1]} pagador"])
        ax.set_xlim(0.5, 2.9)
        ax.set_title(DESCRIPTIONS[col])
        ax.grid(axis="x", visible=False)
    fig.suptitle("Maus pagadores pedem créditos mais longos e maiores, e são um pouco mais jovens",
                 y=1.04, fontsize=11)
    save(fig, figs / "02_boxplot_numericos.png")
    return desc


def section_categorical(df, figs, tables, top):
    """Tabela com todas as categorias; gráfico apenas dos atributos em `top`."""
    overall = df[TARGET].mean() * 100
    rows = []
    for col in CATEGORICAL:
        order = [c for c in ordered_categories(col) if c in df[col].unique()]
        g = df.groupby(col)[TARGET].agg(["mean", "size"]).reindex(order)
        for c, (m, n) in zip(order, zip(g["mean"], g["size"])):
            rows.append({"atributo": col, "código": c, "categoria": CODEBOOK[col][c],
                         "n": int(n), "%_da_base": round(n / len(df) * 100, 1),
                         "%_mau": round(m * 100, 1)})
    freq = pd.DataFrame(rows)
    freq.to_csv(tables / "06_frequencias_categoricos.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, col in zip(axes.flat, top):
        f = freq[freq["atributo"] == col]
        y = np.arange(len(f))
        ax.barh(y, f["%_mau"], color=C_BAD, height=0.6)
        ax.set_yticks(y, [f"{c}  (n={n})" for c, n in zip(f["categoria"], f["n"])], fontsize=8)
        ax.invert_yaxis()
        for yi, v in zip(y, f["%_mau"]):
            ax.text(v + 1, yi, f"{v:.0f}%", va="center", fontsize=8, color=C_INK)
        ax.axvline(overall, ls="--", lw=1, color=C_MUTED)
        ax.set_xlim(0, 75)
        ax.set_title(DESCRIPTIONS[col])
        ax.grid(axis="y", visible=False)
    for ax in axes[-1]:
        ax.set_xlabel("% de maus pagadores na categoria")
    fig.suptitle("Taxa de mau pagador nos 4 atributos categóricos mais relevantes\n"
                 f"(linha tracejada = média geral de {overall:.0f}%)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, figs / "03_taxa_mau_categoricos.png")
    return freq


def section_association(df, figs, tables):
    """Associação de cada atributo com a classe."""
    rows = []
    for col in CATEGORICAL:
        v, p = cramers_v(df[col], df[TARGET])
        rows.append({"atributo": col, "tipo": "categórico", "teste": "qui-quadrado",
                     "p_valor": p, "efeito": v, "medida_efeito": "Cramér's V"})
    for col in NUMERIC:
        a, b = df.loc[df[TARGET] == 0, col], df.loc[df[TARGET] == 1, col]
        _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        r, _ = stats.pointbiserialr(df[TARGET], df[col])
        rows.append({"atributo": col, "tipo": "numérico", "teste": "Mann-Whitney U",
                     "p_valor": p, "efeito": abs(r), "medida_efeito": "r ponto-bisserial (abs)"})
    assoc = pd.DataFrame(rows).set_index("atributo")

    # Informação mútua (categóricos codificados como inteiros, tratados como discretos)
    X = df[FEATURES].copy()
    for col in CATEGORICAL:
        X[col] = X[col].astype("category").cat.codes
    discrete_numeric = ["installment_rate", "residence_since", "existing_credits", "num_dependents"]
    discrete_mask = [c in CATEGORICAL or c in discrete_numeric for c in FEATURES]
    mi = mutual_info_classif(X, df[TARGET], discrete_features=discrete_mask, random_state=42)
    assoc["informação_mútua"] = pd.Series(mi, index=FEATURES)
    assoc["significativo_5%"] = assoc["p_valor"] < 0.05
    assoc = assoc.sort_values("informação_mútua", ascending=False)
    assoc.to_csv(tables / "07_associacao_com_classe.csv")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    y = np.arange(len(assoc))
    ax.barh(y, assoc["informação_mútua"], color=C_GOOD, height=0.65)
    ax.set_yticks(y, [DESCRIPTIONS[c] for c in assoc.index], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("informação mútua com a classe\n(maior = ajuda mais a separar bons de maus pagadores)")
    ax.set_title("Quais atributos mais ajudam a prever o risco?")
    ax.grid(axis="y", visible=False)
    save(fig, figs / "04_relevancia_atributos.png")
    return assoc


def section_correlation(df, figs, tables):
    spear = df[NUMERIC].corr(method="spearman")
    spear.round(3).to_csv(tables / "08_correlacao_spearman.csv")

    mask = np.triu(np.ones_like(spear, dtype=bool))
    mat = np.ma.masked_array(spear.values, mask=mask)
    labels = [DESCRIPTIONS[c].split(" (")[0] for c in NUMERIC]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(len(NUMERIC)):
        for j in range(i):
            v = spear.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.5 else C_INK)
    ax.set_xticks(range(len(NUMERIC) - 1), labels[:-1], rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(1, len(NUMERIC)), labels[1:], fontsize=8)
    ax.set_xlim(-0.5, len(NUMERIC) - 1.5)
    ax.set_ylim(len(NUMERIC) - 0.5, 0.5)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.75, label="correlação de Spearman")
    ax.set_title("Correlação entre atributos numéricos\n(só duração × valor do crédito é forte)")
    save(fig, figs / "05_correlacao_numericos.png")

    n = len(CATEGORICAL)
    cv = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            cv[i, j] = cv[j, i] = cramers_v(df[CATEGORICAL[i]], df[CATEGORICAL[j]])[0]
    cv_df = pd.DataFrame(cv, index=CATEGORICAL, columns=CATEGORICAL)
    cv_df.round(3).to_csv(tables / "09_cramers_v_categoricos.csv")

    pairs = []
    for i in range(len(NUMERIC)):
        for j in range(i + 1, len(NUMERIC)):
            pairs.append((NUMERIC[i], NUMERIC[j], spear.iloc[i, j]))
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((CATEGORICAL[i], CATEGORICAL[j], cv[i, j]))
    strong = sorted([p for p in pairs if abs(p[2]) >= 0.3], key=lambda p: -abs(p[2]))
    return spear, cv_df, strong


def baselines(df):
    """Referência para a modelagem, usando a matriz de custo do UCI (FN=5, FP=1)."""
    n, n_bad = len(df), int(df[TARGET].sum())
    n_good = n - n_bad
    return {
        "tudo 'bom'": {"acurácia": n_good / n, "custo_total": 5 * n_bad, "custo_médio": 5 * n_bad / n},
        "tudo 'mau'": {"acurácia": n_bad / n, "custo_total": 1 * n_good, "custo_médio": n_good / n},
    }


# ----------------------------------------------------------------- relatório
def write_report(out, df, info, dist, q, rare_df, desc, freq, assoc, strong, base):
    chk = q["checagens"]
    ratio = dist.loc["bom", "n"] / dist.loc["mau", "n"]
    L = []
    L.append("# Análise exploratória — Statlog (German Credit Data)\n")
    L.append(f"Fonte: UCI Machine Learning Repository — <{UCI_URL}>  ")
    L.append("Doador: Prof. Dr. Hans Hofmann, Universität Hamburg (1994). Arquivo: `german.data`.\n")

    L.append("## 1. Problema\n")
    L.append("Classificação binária: dado o perfil de um solicitante de crédito, prever se ele é "
             "**bom** ou **mau** pagador. A base vem com uma **matriz de custo**: classificar um mau "
             "pagador como bom custa **5**, classificar um bom como mau custa **1**. Logo, a classe "
             "positiva de interesse é *mau* e o recall dessa classe importa mais que a acurácia.\n")

    L.append("## 2. Visão geral\n")
    L.append(f"- Instâncias: **{len(df)}**")
    L.append(f"- Atributos preditores: **{len(FEATURES)}** ({len(NUMERIC)} numéricos, "
             f"{len(CATEGORICAL)} categóricos)")
    L.append(f"- Atributo classe: `{TARGET}` (original 1=bom, 2=mau; recodificado para 0=bom, 1=mau)\n")
    L.append(md_table(info[["descrição", "tipo", "n_valores_distintos", "ausentes"]]))
    L.append("\n\\* ordinal exceto a categoria \"sem conta\"/\"desconhecido\", que não tem posição natural na escala.\n")

    L.append("## 3. Distribuição da classe\n")
    L.append(md_table(dist, "{:.1f}"))
    acc_trivial = base["tudo 'bom'"]["acurácia"]
    L.append(f"\nRazão bom:mau = **{ratio:.2f}:1** → desbalanceamento moderado. Um classificador que "
             f"responde sempre \"bom\" já tem {acc_trivial:.0%} de acurácia, "
             "por isso acurácia sozinha não serve como métrica. Usar estratificação nas divisões "
             "treino/validação/teste e métricas como F1/recall da classe *mau*, AUC-ROC e custo total.\n")
    L.append("![](figs/01_distribuicao_classes.png)\n")

    L.append("## 4. Qualidade dos dados\n")
    L.append(f"- Valores ausentes: **{q['ausentes_total']}**")
    L.append(f"- Linhas duplicadas: **{q['linhas_duplicadas']}**")
    L.append(f"- Instâncias com mesmos atributos e rótulos diferentes: **{q['features_duplicadas_rotulo_conflitante']}**")
    L.append(f"- Categorias documentadas que nunca aparecem: {', '.join(q['categorias_documentadas_ausentes']) or 'nenhuma'}")
    L.append(f"- Códigos não documentados: {', '.join(q['codigos_nao_documentados']) or 'nenhum'}")
    L.append("- Checagens de domínio/consistência:\n")
    L.append(md_table(pd.DataFrame.from_dict(chk, orient="index", columns=["ocorrências"])
                      .rename_axis("checagem")))
    L.append("\nCategorias raras (< 2% da base) — candidatas a agrupamento:\n")
    L.append(md_table(rare_df.set_index("atributo"), "{:.1f}") if len(rare_df) else "nenhuma")
    L.append("\nObservações:")
    L.append("- `personal_status` mistura sexo e estado civil; não existe a categoria \"mulher solteira\" (A95) "
             "na base, então sexo e estado civil não podem ser separados completamente. Esse atributo "
             "(e `foreign_worker`) é sensível do ponto de vista de justiça algorítmica.")
    L.append("- `foreign_worker` é quase constante (≈96% \"sim\").")
    L.append("- As checagens semânticas acusam casos *suspeitos*, mas não necessariamente erros: "
             "`employment`=desempregado com `job` qualificado pode indicar a qualificação profissional "
             "de alguém hoje sem emprego; A30 (\"sem créditos / **todos pagos**\") é ambígua no próprio "
             "dicionário, então ter mais de um crédito não a contradiz. Optamos por **não remover** essas linhas.")
    L.append("- `num_dependents` tem IQR = 0 (85% das linhas valem 1), por isso todos os valores 2 "
             "aparecem como \"outliers\" — artefato do método, não erro.\n")

    L.append("## 5. Atributos numéricos\n")
    cols = ["mean", "std", "min", "50%", "max", "assimetria", "outliers_IQR", "média_bom", "média_mau"]
    L.append(md_table(desc[cols], "{:.2f}"))
    L.append("\n- `credit_amount` tem forte assimetria à direita (cauda longa); `duration` e `age` também "
             "são assimétricas. Transformação log e/ou padronização ajudam modelos sensíveis à escala "
             "(MLP, KNN, SVM, regressão logística).")
    L.append("- Escalas muito diferentes (idade ~20-75 vs. valor até ~18 mil DM) → normalizar é obrigatório para a MLP.")
    L.append("- Os outliers por IQR são valores plausíveis (créditos altos e longos), não erros; não devem ser removidos.\n")
    L.append("![](figs/02_boxplot_numericos.png)\n")

    L.append("## 6. Atributos categóricos\n")
    L.append("Gráfico dos 4 atributos categóricos mais associados à classe (os demais estão em "
             "`tables/06_frequencias_categoricos.csv`):\n")
    L.append("![](figs/03_taxa_mau_categoricos.png)\n")
    top_cat = freq.sort_values("%_mau", ascending=False)
    top_cat = top_cat[top_cat["n"] >= 30].head(6)
    L.append("Categorias com maior taxa de mau pagador (n ≥ 30):\n")
    L.append(md_table(top_cat.set_index("atributo")[["categoria", "n", "%_mau"]], "{:.1f}"))
    L.append("")

    L.append("## 7. Associação com a classe\n")
    L.append(md_table(assoc[["tipo", "teste", "p_valor", "medida_efeito", "efeito", "informação_mútua"]], "{:.4f}"))
    sig = assoc[assoc["significativo_5%"]].index.tolist()
    nsig = assoc[~assoc["significativo_5%"]].index.tolist()
    L.append(f"\n- Significativos a 5% ({len(sig)}): {', '.join(sig)}")
    L.append(f"- Não significativos ({len(nsig)}): {', '.join(nsig)}")
    L.append("- Na informação mútua, os numéricos discretos (`installment_rate`, `residence_since`, "
             "`existing_credits`, `num_dependents`) são tratados como discretos; os contínuos usam o "
             "estimador por k-vizinhos.")
    L.append("- Nenhum atributo isolado separa bem as classes (maior Cramér's V ≈ 0,35): o problema é "
             "difícil e o ganho deve vir da combinação de atributos.\n")
    L.append("![](figs/04_relevancia_atributos.png)\n")

    L.append("## 8. Correlações entre atributos\n")
    if strong:
        L.append("Pares com |ρ| ou V ≥ 0,3:\n")
        for a, b, v in strong:
            L.append(f"- `{a}` × `{b}`: {v:.2f}")
    L.append("\n![](figs/05_correlacao_numericos.png)\n")
    L.append("Associação entre categóricos (Cramér's V) em `tables/09_cramers_v_categoricos.csv`.\n")

    L.append("## 9. Referências para a modelagem (matriz de custo FN=5, FP=1)\n")
    L.append(md_table(pd.DataFrame(base).T.rename_axis("classificador trivial"), "{:.3f}"))
    L.append("")

    L.append("## 10. Implicações para o pré-processamento\n")
    L.append("1. **Sem ausentes/duplicatas** → não é preciso imputação nem remoção.")
    L.append("2. **Categóricos nominais** → one-hot encoding (MLP, KNN, SVM, reg. logística). Árvores podem usar códigos inteiros.")
    L.append("3. **Categóricos ordinais** (`employment`, `job`, `checking_status`, `savings_status`) → podem ser "
             "codificados ordinalmente, mas \"sem conta\"/\"desconhecido\" quebra a ordem; one-hot é a opção mais segura.")
    L.append("4. **Numéricos** → padronização (z-score); considerar log1p em `credit_amount` (e `duration`).")
    L.append("5. **Categorias raras** (ex.: `purpose` = requalificação/eletrodomésticos) → agrupar em \"outros\" para reduzir dimensionalidade e ruído no one-hot.")
    L.append("6. **Desbalanceamento 70/30 + custo assimétrico** → divisão estratificada treino/validação/teste, "
             "pesos de classe na loss (`weight_bias` na config) ou ajuste do limiar de decisão na validação; "
             "avaliar com recall/F1 da classe mau, AUC e custo.")
    L.append("7. **Base pequena (1000 linhas)** → MLP pequena, regularização (dropout/weight decay) e parada antecipada. "
             "Com conjuntos de validação/teste pequenos, repetir com várias seeds de divisão para medir a variabilidade.")
    L.append("8. Ajustar o pré-processamento (scaler, encoder) **apenas no treino**; validação e teste só recebem "
             "`transform`, para evitar vazamento de dados.")

    (out / "relatorio_eda.md").write_text("\n".join(L), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(PROJECT_ROOT / "eda" / "outputs"), help="pasta de saída")
    args = parser.parse_args()

    out = Path(args.out)
    figs, tables = out / "figs", out / "tables"
    if figs.exists():
        shutil.rmtree(figs)  # remove gráficos de execuções anteriores
    figs.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    df = load_raw()
    df_dec = load_raw(decode=True)
    df_dec.to_csv(tables / "00_base_decodificada.csv", index=False)

    print(f"Base carregada: {df.shape[0]} instâncias × {df.shape[1] - 1} atributos + classe")
    info = section_overview(df, tables)
    dist = section_target(df, figs, tables)
    q, rare_df = section_quality(df, tables)
    desc = section_numeric(df, figs, tables)
    assoc = section_association(df, figs, tables)
    top_cat = [c for c in assoc.index if c in CATEGORICAL][:4]
    freq = section_categorical(df, figs, tables, top_cat)
    _, _, strong = section_correlation(df, figs, tables)
    base = baselines(df)
    write_report(out, df, info, dist, q, rare_df, desc, freq, assoc, strong, base)

    print("\n=== Resumo ===")
    print(dist.to_string())
    print(f"Ausentes: {q['ausentes_total']} | Duplicatas: {q['linhas_duplicadas']}")
    print(f"Categorias documentadas ausentes: {q['categorias_documentadas_ausentes']}")
    print("Checagens:", q["checagens"])
    print("\nTop 8 atributos por informação mútua:")
    print(assoc[["informação_mútua", "p_valor"]].head(8).round(4).to_string())
    print("\nPares fortemente associados (>=0.3):", [(a, b, round(v, 2)) for a, b, v in strong])
    print(f"\nSaídas em: {out.resolve()}")


if __name__ == "__main__":
    main()
