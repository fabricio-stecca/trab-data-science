# Etapa 1 — Análise exploratória

Cobre o item 2a do enunciado: problema, fonte, instâncias e atributos, classe e seu
balanceamento, qualidade dos dados, gráficos e tabelas.

## Como rodar

```bash
python eda/run_eda.py                    # a partir da raiz do projeto
python eda/run_eda.py --out outra/pasta  # opcional: outra pasta de saída
```

## O que é gerado (`eda/outputs/`)

**`relatorio_eda.md`** — resumo em texto com as tabelas e as conclusões, organizado nas seções
que o enunciado pede (problema, visão geral, classes, qualidade, numéricos, categóricos,
associação com a classe, correlações e implicações para o pré-processamento).

**`figs/`** — 5 gráficos, um por pergunta:

| arquivo | pergunta que responde |
|---|---|
| `01_distribuicao_classes.png` | Quantos bons e maus pagadores existem? |
| `02_boxplot_numericos.png` | Duração, valor e idade diferem entre bons e maus? |
| `03_taxa_mau_categoricos.png` | Nos 4 atributos categóricos mais relevantes, quais categorias têm mais maus pagadores? |
| `04_relevancia_atributos.png` | Quais atributos mais ajudam a prever o risco? |
| `05_correlacao_numericos.png` | Os atributos numéricos são redundantes entre si? |

**`tables/`** — tabelas completas em CSV:

| arquivo | conteúdo |
|---|---|
| `00_base_decodificada.csv` | base com os códigos (A11, A12…) trocados por rótulos legíveis |
| `01_atributos.csv` | nome, descrição, tipo, nº de valores distintos e ausentes |
| `02_distribuicao_classes.csv` | contagem e % de cada classe |
| `03_categorias_raras.csv` | categorias com menos de 2% da base |
| `04_checagens_consistencia.csv` | checagens de domínio e de consistência entre atributos |
| `05_estatisticas_numericos.csv` | estatísticas descritivas, assimetria, outliers e médias por classe |
| `06_frequencias_categoricos.csv` | frequência e taxa de mau pagador de todas as categorias |
| `07_associacao_com_classe.csv` | qui-quadrado/Cramér's V, Mann-Whitney e informação mútua |
| `08_correlacao_spearman.csv` | correlação entre numéricos |
| `09_cramers_v_categoricos.csv` | associação entre categóricos |

## Principais achados

- **Classes desbalanceadas (70/30)**: um modelo que diz sempre "bom" acerta 70%, então
  acurácia sozinha não serve; usar recall/F1 da classe *mau*, AUC e custo.
- **Sem valores ausentes nem duplicatas.** Duas categorias documentadas nunca aparecem
  (finalidade "férias" e "mulher solteira").
- **Atributos mais associados à classe**: situação da conta corrente, duração, histórico de
  crédito, poupança, idade e finalidade. Nenhum separa bem as classes sozinho.
- **Escalas muito diferentes e assimetria** (valor do crédito vai até ~18 mil DM) → padronizar e
  aplicar log no valor e na duração.
- Único par numérico fortemente correlacionado: duração × valor do crédito (ρ = 0,62).

## Observação

A análise exploratória usa a base inteira, por ser descritiva. Decisões de pré-processamento
tiradas daqui (ex.: aplicar log em `credit_amount`) são escolhas de projeto; os parâmetros
dessas transformações (médias, desvios, categorias raras) são aprendidos só no treino, na etapa 2.
