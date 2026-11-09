"""Leitura dos YAMLs: herança, overrides via linha de comando e sweeps.

Ordem de herança (cada nível sobrescreve o anterior):
    common/protocolo.yaml  ->  <algoritmo>/configs/default.yaml  ->  arquivo do experimento  ->  --set
"""
import copy
import glob
import hashlib
import itertools
import re
from pathlib import Path

import yaml

from common import PROJECT_ROOT

PROTOCOL_CONFIG = PROJECT_ROOT / "common" / "protocolo.yaml"


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def check_keys(default: dict, user: dict, prefix: str = ""):
    """Erro se o YAML do usuário tiver chaves que não existem no default (ex.: opção removida)."""
    for k, v in (user or {}).items():
        if not prefix and k == "sweep":
            continue
        path = f"{prefix}{k}"
        if k not in default:
            raise KeyError(f"Chave desconhecida na config: '{path}' "
                           "(não existe no default do algoritmo nem em common/protocolo.yaml)")
        if isinstance(v, dict) and isinstance(default[k], dict):
            check_keys(default[k], v, path + ".")


def set_by_path(cfg: dict, dotted: str, value):
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            raise KeyError(f"Chave inexistente na config: '{dotted}'")
        node = node[k]
    if keys[-1] not in node:
        raise KeyError(f"Chave inexistente na config: '{dotted}'")
    node[keys[-1]] = value


def parse_overrides(items) -> dict:
    """['model.activation=tanh', 'model.hidden_layers=[64,32]'] -> {chave: valor}"""
    out = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Override inválido '{item}'. Use chave.subchave=valor")
        k, v = item.split("=", 1)
        out[k.strip()] = yaml.safe_load(v)
    return out


def _label(key: str, value) -> str:
    parts = key.split(".")
    short = parts[-2] if parts[-1] == "name" and len(parts) > 1 else parts[-1]
    if isinstance(value, list):
        value = "-".join(map(str, value))
    # só caracteres válidos em nome de pasta
    return re.sub(r"[^\w.=-]", "_", f"{short}={value}")[:40]


def _auto_name(base: str, overrides: dict) -> str:
    labels = [_label(k, v) for k, v in overrides.items() if not k.startswith("experiment.")]
    name = "__".join([base] + labels)
    if len(name) > 80:  # Windows limita caminhos a 260 caracteres
        digest = hashlib.md5(name.encode()).hexdigest()[:6]
        name = f"{name[:73]}_{digest}"
    return name


def expand_sweep(cfg: dict) -> list[dict]:
    """Bloco `sweep:` {chave.pontuada: [valores]} -> produto cartesiano de configs."""
    sweep = cfg.pop("sweep", None)
    if not sweep:
        return [cfg]
    keys = list(sweep)
    configs = []
    for combo in itertools.product(*(sweep[k] for k in keys)):
        c = copy.deepcopy(cfg)
        for k, v in zip(keys, combo):
            set_by_path(c, k, v)
        c["experiment"]["name"] = _auto_name(cfg["experiment"]["name"], dict(zip(keys, combo)))
        configs.append(c)
    return configs


def _read_yaml(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_configs(paths, overrides, default_config: Path) -> list[dict]:
    """Carrega um ou mais YAMLs (aceita curingas), herda protocolo + default do algoritmo
    e aplica os overrides."""
    protocol = _read_yaml(PROTOCOL_CONFIG)
    algo_default = _read_yaml(default_config)
    for sec in ("split", "evaluation"):
        if sec in algo_default:
            raise KeyError(f"'{sec}' deve ficar só em common/protocolo.yaml (igual para todos os algoritmos)")
    default = deep_merge(protocol, algo_default)

    files = []
    for p in paths or [str(default_config)]:
        matches = sorted(glob.glob(p)) or [p]
        files.extend(matches)

    ov = parse_overrides(overrides)
    result = []
    for path in files:
        user = _read_yaml(path)
        check_keys(default, user)
        cfg = deep_merge(default, user)
        for k, v in ov.items():
            set_by_path(cfg, k, v)
        if ov and Path(path).resolve() == Path(default_config).resolve() and "experiment.name" not in ov:
            cfg["experiment"]["name"] = _auto_name(cfg["experiment"]["name"], ov)
        result.extend(expand_sweep(cfg))
    return result
