# KNN — k vizinhos mais próximos

Implementação: `KNeighborsClassifier` (scikit-learn). Segue o protocolo comum
([README da raiz](../README.md#protocolo-de-avaliação-igual-para-todos-os-algoritmos)).

## Como funciona

Para classificar um cliente, calcula a distância dele a todos os clientes do treino, pega os
`k` mais próximos e usa a proporção de maus pagadores entre eles como P(mau). Com
`weights: distance`, vizinhos mais próximos pesam mais. Não há treino propriamente dito: o
modelo só guarda o conjunto de treino.

## Como rodar

```bash
python knn/train.py                                       # configs/default.yaml (k=15, distance)
python knn/train.py -c knn/configs/experiments/grade.yaml # parametrizações testadas
python knn/train.py --set model.n_neighbors=7
```

## Parâmetros (`configs/default.yaml`)

| parâmetro | significado | testado em `grade.yaml` |
|---|---|---|
| `n_neighbors` | k: nº de vizinhos consultados (pequeno = sensível a ruído; grande = mais suave) | 5, 15, 31 |
| `weights` | `uniform` (votos iguais) ou `distance` (mais próximos pesam mais) | uniform, distance |
| `p` | 1 = distância de Manhattan, 2 = Euclidiana | 2 |

## Pré-processamento

- **Log + padronização obrigatórios**: a distância somaria diretamente idade (~20-75) e valor do
  crédito (até ~18 mil), e o valor dominaria tudo.
- One-hot nos categóricos (distância entre categorias = 0 ou 1).
- **Desbalanceamento**: o KNN não aceita peso de classe; é tratado pelo limiar escolhido na
  validação.
