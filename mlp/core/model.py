"""MLP configurável."""
import torch.nn as nn

ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "selu": nn.SELU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


class MLP(nn.Module):
    """Linear -> [BatchNorm] -> ativação -> [Dropout], repetido por camada oculta,
    seguido de uma camada linear de saída (logits)."""

    def __init__(self, in_features: int, out_features: int, hidden_layers: list[int],
                 activation: str = "relu", dropout: float = 0.0, batch_norm: bool = False,
                 weight_init: str = "default"):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"ativação '{activation}' inválida. Opções: {list(ACTIVATIONS)}")
        layers, prev = [], in_features
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(ACTIVATIONS[activation]())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_features))
        self.net = nn.Sequential(*layers)
        self._init_weights(weight_init, activation)

    def _init_weights(self, scheme: str, activation: str):
        if scheme == "default":
            return
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if scheme == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                elif scheme == "kaiming":
                    nonlin = "leaky_relu" if activation == "leaky_relu" else "relu"
                    nn.init.kaiming_uniform_(m.weight, nonlinearity=nonlin)
                else:
                    raise ValueError(f"weight_init '{scheme}' inválido (default | xavier | kaiming)")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


def build_model(model_cfg: dict, in_features: int, out_features: int) -> MLP:
    return MLP(
        in_features=in_features,
        out_features=out_features,
        hidden_layers=model_cfg["hidden_layers"],
        activation=model_cfg["activation"],
        dropout=model_cfg.get("dropout", 0.0),
        batch_norm=model_cfg.get("batch_norm", False),
        weight_init=model_cfg.get("weight_init", "default"),
    )
