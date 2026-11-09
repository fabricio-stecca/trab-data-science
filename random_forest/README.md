# Random Forest

Implementação: `RandomForestClassifier` (scikit-learn). Segue o protocolo comum
([README da raiz](../README.md#protocolo-de-avaliação-igual-para-todos-os-algoritmos)).

## Como funciona

Treina várias árvores de decisão independentes. Cada árvore usa uma amostra bootstrap do treino
(sorteio com reposição) e, a cada divisão, só pode escolher entre um subconjunto aleatório de
atributos. Isso torna as árvores diferentes entre si; a média das previsões delas (P(mau)) é
mais estável que uma árvore sozinha.

```
para b = 1..n_estimators:
    D_b ← amostra bootstrap do treino
    árvore_b ← árvore de decisão treinada em D_b, sorteando max_features atributos a cada nó
P(mau | x) = média de P_b(mau | x) nas árvores
```

## Como rodar

```bash
python random_forest/train.py
python random_forest/train.py -c random_forest/configs/experiments/grade.yaml
python random_forest/train.py --set model.max_depth=6
```

## Parâmetros (`configs/default.yaml`)

| parâmetro | significado | testado em `grade.yaml` |
|---|---|---|
| `n_estimators` | nº de árvores | 300 |
| `max_depth` | profundidade máxima de cada árvore (`null` = sem limite) | sem limite, 8 |
| `min_samples_leaf` | mínimo de clientes por folha (maior = árvores mais simples) | 1, 5 |
| `max_features` | atributos sorteados a cada divisão | sqrt |
| `weight_bias` | peso da classe "mau" (`auto` = n_bons/n_maus no treino) | auto |

## Pré-processamento

- **Sem log e sem padronização**: árvores só comparam valores com limiares, então a escala não
  muda o resultado.
- One-hot nos categóricos.
- **Desbalanceamento**: `weight_bias` (vira `class_weight` do scikit-learn) + limiar escolhido
  na validação.
