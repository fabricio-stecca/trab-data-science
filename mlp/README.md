# MLP — rede neural multicamadas (PyTorch)

Rede neural configurável por YAML, para testar várias abordagens trocando só a configuração.
Segue o protocolo comum ([README da raiz](../README.md#protocolo-de-avaliação-igual-para-todos-os-algoritmos)):
divisão, limiar e métricas vêm de `common/protocolo.yaml`.

## Como rodar

A partir da raiz do projeto:

```bash
python mlp/train.py                                          # usa mlp/configs/default.yaml
python mlp/train.py -c mlp/configs/experiments/tanh.yaml     # um experimento
python mlp/train.py -c "mlp/configs/experiments/*.yaml"      # vários experimentos
python mlp/train.py -c mlp/configs/sweep_exemplo.yaml        # grade de combinações
python mlp/train.py --set device=cuda model.hidden_layers=[64,32] training.optimizer.lr=0.005
```

## Código

```
mlp/
├── train.py            define como a MLP é treinada; o resto é o protocolo comum (common/runner.py)
├── configs/
│   ├── default.yaml    todas as opções da MLP, comentadas
│   ├── experiments/    seus experimentos (variações do default)
│   └── sweep_exemplo.yaml
└── core/
    ├── model.py        a MLP
    ├── factory.py      otimizador, scheduler e loss (com weight_bias)
    ├── trainer.py      loop de treino, avaliação periódica, checkpoint, parada antecipada
    └── plots.py        curvas de aprendizado
```

## Opções da configuração (`configs/default.yaml`)

| seção | opções |
|---|---|
| `device` | `cpu`, `cuda`, `auto` — nesta base a CPU é mais rápida (rede e batches pequenos) |
| `data` | `drop_features`, `log_transform`, `scaler` (`standard`/`minmax`/`robust`/`none`), `rare_category_min_frequency` |
| `model` | `hidden_layers`, `activation` (`relu`, `leaky_relu`, `elu`, `gelu`, `selu`, `silu`, `tanh`, `sigmoid`), `dropout`, `batch_norm`, `weight_init` (`default`/`xavier`/`kaiming`) |
| `training` | `epochs`, `batch_size`, `grad_clip`, `eval_every_steps`, `monitor` (`val_loss`/`val_auc`/`val_f1`/`val_cost`) |
| `training.optimizer` | `name` (`adam`/`adamw`), `lr`, `weight_decay` |
| `training.scheduler` | `name` (`none`/`step`/`cosine`), `step_size`, `gamma` |
| `training.loss` | `name` (`bce`/`cross_entropy`/`mse`), `weight_bias` (`auto`/número/`null`) |
| `training.early_stopping` | `enabled`, `patience` (nº de avaliações sem melhora), `min_delta` |

`split` e `evaluation` (limiar, custos, métrica de escolha) ficam em `common/protocolo.yaml`,
iguais para todos os algoritmos; podem ser sobrescritos com `--set`, mas aí o `comparar.py`
ignora a execução se a divisão for diferente.

### Desbalanceamento de classes

- **`training.loss.weight_bias`**: peso da classe *mau* na loss. `auto` = n_bons / n_maus no
  treino (≈ 2,33); um número fixa o peso; `null` desliga.
- **Limiar** (`evaluation.threshold` no protocolo): `optimize_cost` escolhe, na validação, o
  limiar que minimiza 5·FN + 1·FP.

## Treino: checkpoint e parada antecipada

A cada `eval_every_steps` atualizações de pesos o modelo é avaliado na **validação**. Sempre
que `training.monitor` melhora, o checkpoint é salvo em `best_model.pt`. Se não melhorar por
`early_stopping.patience` avaliações seguidas, o treino para. No final, o melhor checkpoint é
carregado e usado para escolher o limiar (validação) e avaliar o teste.

## Criando experimentos

Crie arquivos em `configs/experiments/` contendo **só o que muda** em relação ao
`default.yaml`. Prefira mudar **uma coisa por experimento**, para poder atribuir a diferença
de resultado a uma causa. Exemplo (`configs/experiments/tanh.yaml`):

```yaml
experiment:
  name: tanh
model:
  activation: tanh
```

## Saídas de cada execução (`resultados/runs/mlp/<nome>_<data-hora>/`)

| arquivo | conteúdo |
|---|---|
| `config.yaml` | cópia exata da configuração usada |
| `split.csv` | índice de cada linha e o conjunto (treino/validacao/teste) |
| `best_model.pt` | melhor checkpoint: pesos, step/época, score, métricas de validação e config |
| `preprocessador.joblib` | pré-processador ajustado no treino (necessário para usar o checkpoint) |
| `historico_treino.csv` | loss de treino e métricas de validação a cada avaliação |
| `metricas.json` | limiar, métricas de validação e de teste |
| `predicoes_teste.csv` | probabilidade e predição para cada linha de teste |
| `curvas_aprendizado.png`, `matriz_confusao.png`, `curva_roc.png` | gráficos |

Toda execução também acrescenta uma linha em `resultados/resultados.csv`.

### Carregar um checkpoint

```python
import sys, json, joblib, torch
sys.path.insert(0, ".")  # raiz do projeto
from mlp.core.model import build_model

pasta = "resultados/runs/mlp/<pasta>"
ckpt = torch.load(f"{pasta}/best_model.pt", weights_only=True)
model = build_model(ckpt["config"]["model"], ckpt["in_features"], ckpt["n_outputs"])
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
prep = joblib.load(f"{pasta}/preprocessador.joblib")   # prep.transform(df_com_os_atributos)
limiar = json.load(open(f"{pasta}/metricas.json", encoding="utf-8"))["threshold"]
```
