# Gradient Boosting

Implementação: `HistGradientBoostingClassifier` (scikit-learn). Segue o protocolo comum
([README da raiz](../README.md#protocolo-de-avaliação-igual-para-todos-os-algoritmos)).

## Como funciona

Treina árvores pequenas **em sequência**. Cada nova árvore é ajustada ao gradiente da loss
(log-loss) do modelo atual, ou seja, aprende a corrigir os erros das árvores anteriores. A
contribuição de cada árvore é reduzida pela `learning_rate`. A versão "Hist" agrupa os valores
numéricos em faixas (histogramas), o que a torna rápida.

```
F_0(x) ← log(proporção de maus / proporção de bons)          # previsão inicial
para m = 1..max_iter:
    r_i ← y_i − sigmoide(F_{m−1}(x_i))                         # erros (gradiente negativo)
    árvore_m ← árvore pequena ajustada para prever r_i
    F_m(x) ← F_{m−1}(x) + learning_rate · árvore_m(x)
P(mau | x) = sigmoide(F_M(x))
```

## Como rodar

```bash
python gradient_boosting/train.py
python gradient_boosting/train.py -c gradient_boosting/configs/experiments/grade.yaml
python gradient_boosting/train.py --set model.learning_rate=0.05
```

## Parâmetros (`configs/default.yaml`)

| parâmetro | significado | testado em `grade.yaml` |
|---|---|---|
| `learning_rate` | peso de cada nova árvore (menor = aprende devagar, mais estável) | 0.03, 0.1 |
| `max_iter` | nº de árvores | 200 |
| `max_depth` | profundidade máxima de cada árvore (`null` = limitada por `max_leaf_nodes`) | 3, sem limite |
| `max_leaf_nodes` | nº máximo de folhas por árvore | 31 |
| `min_samples_leaf` | mínimo de clientes por folha | 20 |
| `l2_regularization` | penalização L2 nos valores das folhas | 0 |
| `weight_bias` | peso da classe "mau" (`auto` = n_bons/n_maus no treino) | auto |

O early stopping interno do scikit-learn fica desligado (`max_iter` fixo): a escolha entre
configurações é feita na validação do protocolo comum.

## Pré-processamento

- **Sem log e sem padronização** (baseado em árvores).
- One-hot nos categóricos.
- **Desbalanceamento**: `weight_bias` (vira `class_weight`) + limiar escolhido na validação.
