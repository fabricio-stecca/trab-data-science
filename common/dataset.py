"""Carregamento e metadados do Statlog (German Credit Data) - UCI id 144.

Fonte: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
Arquivo usado: german.data (20 atributos originais, 7 numéricos + 13 categóricos).
"""
from pathlib import Path

import pandas as pd

from common import PROJECT_ROOT

RAW_FILE = PROJECT_ROOT / "data" / "raw" / "german.data"
UCI_URL = "https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data"

TARGET = "credit_risk"
# Classe original: 1 = bom pagador, 2 = mau pagador.
# Recodificamos para 0 = bom, 1 = mau (classe positiva = mau, a de maior custo).
TARGET_LABELS = {0: "bom", 1: "mau"}

# (nome, tipo, descrição)
ATTRIBUTES = [
    ("checking_status", "categórico ordinal*", "Situação da conta corrente"),
    ("duration", "numérico", "Duração do crédito (meses)"),
    ("credit_history", "categórico nominal", "Histórico de crédito"),
    ("purpose", "categórico nominal", "Finalidade do crédito"),
    ("credit_amount", "numérico", "Valor do crédito (DM)"),
    ("savings_status", "categórico ordinal*", "Poupança / títulos"),
    ("employment", "categórico ordinal", "Tempo no emprego atual"),
    ("installment_rate", "numérico discreto", "Parcela em % da renda disponível (1-4)"),
    ("personal_status", "categórico nominal", "Estado civil e sexo"),
    ("other_parties", "categórico nominal", "Outros devedores / fiadores"),
    ("residence_since", "numérico discreto", "Tempo na residência atual (1-4)"),
    ("property", "categórico nominal", "Bens / propriedades"),
    ("age", "numérico", "Idade (anos)"),
    ("other_installment_plans", "categórico nominal", "Outros parcelamentos"),
    ("housing", "categórico nominal", "Moradia"),
    ("existing_credits", "numérico discreto", "Nº de créditos existentes neste banco"),
    ("job", "categórico ordinal", "Tipo de emprego / qualificação"),
    ("num_dependents", "numérico discreto", "Nº de dependentes"),
    ("own_telephone", "categórico nominal", "Telefone em nome do cliente"),
    ("foreign_worker", "categórico nominal", "Trabalhador estrangeiro"),
]
# * ordinal exceto pela categoria "sem conta"/"desconhecido".

FEATURES = [a[0] for a in ATTRIBUTES]
NUMERIC = [a[0] for a in ATTRIBUTES if a[1].startswith("numérico")]
CATEGORICAL = [a[0] for a in ATTRIBUTES if a[1].startswith("categórico")]
DESCRIPTIONS = {a[0]: a[2] for a in ATTRIBUTES}
TYPES = {a[0]: a[1] for a in ATTRIBUTES}

# Decodificação dos códigos simbólicos (german.doc)
CODEBOOK = {
    "checking_status": {"A11": "< 0 DM", "A12": "0-200 DM", "A13": ">= 200 DM", "A14": "sem conta"},
    "credit_history": {
        "A30": "sem créditos/todos pagos",
        "A31": "todos pagos neste banco",
        "A32": "pagos em dia até agora",
        "A33": "atrasos no passado",
        "A34": "conta crítica/outros créditos",
    },
    "purpose": {
        "A40": "carro (novo)", "A41": "carro (usado)", "A42": "móveis/equip.",
        "A43": "rádio/TV", "A44": "eletrodomésticos", "A45": "reparos",
        "A46": "educação", "A47": "férias", "A48": "requalificação",
        "A49": "negócios", "A410": "outros",
    },
    "savings_status": {
        "A61": "< 100 DM", "A62": "100-500 DM", "A63": "500-1000 DM",
        "A64": ">= 1000 DM", "A65": "desconhecido/sem poupança",
    },
    "employment": {
        "A71": "desempregado", "A72": "< 1 ano", "A73": "1-4 anos",
        "A74": "4-7 anos", "A75": ">= 7 anos",
    },
    "personal_status": {
        "A91": "M divorciado/separado", "A92": "F div./sep./casada",
        "A93": "M solteiro", "A94": "M casado/viúvo", "A95": "F solteira",
    },
    "other_parties": {"A101": "nenhum", "A102": "co-requerente", "A103": "fiador"},
    "property": {
        "A121": "imóvel", "A122": "poupança/seguro de vida",
        "A123": "carro/outros", "A124": "desconhecido/nenhum",
    },
    "other_installment_plans": {"A141": "banco", "A142": "lojas", "A143": "nenhum"},
    "housing": {"A151": "aluguel", "A152": "própria", "A153": "gratuita"},
    "job": {
        "A171": "desempregado/não qualif. (não residente)",
        "A172": "não qualificado (residente)",
        "A173": "qualificado/funcionário",
        "A174": "gerência/autônomo/alta qualif.",
    },
    "own_telephone": {"A191": "não", "A192": "sim"},
    "foreign_worker": {"A201": "sim", "A202": "não"},
}


def load_raw(path: Path | str = RAW_FILE, decode: bool = False) -> pd.DataFrame:
    """Lê german.data. Retorna DataFrame com 20 atributos + coluna alvo (0=bom, 1=mau).

    decode=True troca os códigos A11, A12... pelos rótulos legíveis.
    """
    df = pd.read_csv(path, sep=r"\s+", header=None, names=FEATURES + [TARGET])
    df[TARGET] = df[TARGET].map({1: 0, 2: 1}).astype(int)
    if decode:
        for col, mapping in CODEBOOK.items():
            df[col] = df[col].map(mapping).fillna(df[col])
    return df
