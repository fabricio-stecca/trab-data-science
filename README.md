# Trabalho Final — Ciência de Dados I (UFU)

Classificação de risco de crédito no **Statlog (German Credit Data)**: dado o perfil de um
solicitante, prever se ele será **bom** ou **mau** pagador.

- Fonte: UCI Machine Learning Repository, id 144 —
  <https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data>
- 1000 clientes, 20 atributos (7 numéricos, 13 categóricos), 70% bons / 30% maus.
- A base define uma matriz de custo: aprovar um mau pagador custa **5**, negar um bom custa **1**.
- Enunciado do trabalho: [docs/Trabalho-CD.pdf](docs/Trabalho-CD.pdf)

## Estrutura

```
.
├── data/raw/            base original da UCI
├── common/              código compartilhado + PROTOCOLO DE AVALIAÇÃO comum a todos os algoritmos
│   ├── protocolo.yaml   divisão treino/val/teste, limiar e métricas (iguais para todos)
│   ├── runner.py        executa o protocolo: divide, pré-processa, treina, avalia, salva
│   └── ...              dataset, pré-processamento, métricas, gráficos, configs
├── eda/                 1. análise exploratória                    → eda/outputs/
├── mlp/                 2. MLP (PyTorch)
├── knn/                 3. KNN (scikit-learn)
├── random_forest/       4. Random Forest (scikit-learn)
├── gradient_boosting/   5. Gradient Boosting (scikit-learn)
├── comparar.py          6. comparação final entre os algoritmos   → resultados/comparacao/
├── resultados/          saídas de todos os algoritmos (gerado)
│   ├── resultados.csv   uma linha por execução: parâmetros + métricas de validação e teste
│   ├── runs/<algoritmo>/<configuração>_<data-hora>/
│   └── comparacao/
└── docs/                enunciado
```

Cada pasta de algoritmo tem o mesmo formato: `train.py`, `configs/default.yaml`,
`configs/experiments/` e um `README.md`.

## Como rodar

Todos os comandos a partir da raiz do projeto:

```bash
pip install -r requirements.txt

python eda/run_eda.py                                                        # 1. EDA

python mlp/train.py -c mlp/configs/sweep_exemplo.yaml                        # 2. MLP
python knn/train.py -c knn/configs/experiments/grade.yaml                    # 3. KNN
python random_forest/train.py -c random_forest/configs/experiments/grade.yaml          # 4. RF
python gradient_boosting/train.py -c gradient_boosting/configs/experiments/grade.yaml  # 5. GB

python comparar.py                                                           # 6. comparação
```

- `-c` aceita um ou mais YAMLs (e curingas); sem `-c` roda o `default.yaml` do algoritmo.
- `--set chave.subchave=valor` sobrescreve qualquer valor (ex.: `--set model.n_neighbors=7`).
- Um bloco `sweep:` no YAML gera todas as combinações de parâmetros listadas.
- Chaves que não existem nas configs geram erro (evita erro de digitação passar despercebido).

## Protocolo de avaliação (igual para todos os algoritmos)

Definido em [common/protocolo.yaml](common/protocolo.yaml) e executado por
[common/runner.py](common/runner.py). Cada algoritmo só define como treina e como calcula P(mau).

1. **Divisão**: as 1000 linhas são sorteadas em treino / validação / teste (60/20/20,
   estratificado, `split.seed: 42`). Todos os algoritmos usam **exatamente as mesmas linhas**;
   cada execução registra um `split_id` e o `comparar.py` só compara execuções com o mesmo id.
2. **Pré-processamento** (log, escala, one-hot, agrupamento de categorias raras) ajustado
   **só no treino**; validação e teste recebem apenas `transform`. O peso de classe
   (`weight_bias: auto` = n_bons/n_maus) também usa só o treino.
3. **Treino**: o teste nunca é passado ao algoritmo. A MLP usa a validação para escolher o
   checkpoint e parar cedo; os modelos do scikit-learn não usam a validação no ajuste.
4. **Limiar de decisão** (P(mau) ≥ limiar → "mau"): escolhido **só na validação**, minimizando
   o custo 5·FN + 1·FP (`threshold: optimize_cost`).
5. **Escolha da configuração**: para cada algoritmo, a melhor parametrização é escolhida pela
   **validação** (`main_metric: cost_mean`).
6. **Teste**: usado uma única vez por execução, só para relatar o desempenho.

**Métricas** (classe positiva = mau): acurácia, acurácia balanceada, precisão, recall,
especificidade, F1, AUC-ROC e custo (total e médio). Fórmulas em
[common/metrics.py](common/metrics.py).

**Limitações**: validação e teste têm 200 linhas (~60 maus), então os números variam com a
divisão (em 30 divisões diferentes a AUC da MLP variou com desvio de ~0,03). Como o limiar e a
configuração são escolhidos na validação, as métricas de validação são otimistas — o desempenho
a relatar é o do teste. Para medir a variação, repita com outros valores de `split.seed`.

## Status

- [x] Análise exploratória
- [x] MLP (PyTorch), KNN, Random Forest e Gradient Boosting, com várias parametrizações cada
- [x] Comparação entre algoritmos (`comparar.py`)
- [ ] Relatório em PDF (inclui descrição/pseudocódigo dos algoritmos não vistos em sala e a
      declaração de uso de IA generativa exigida no enunciado)
