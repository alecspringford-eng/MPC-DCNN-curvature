from __future__ import annotations

from typing import Any

import yaml

from dcnn_curvature.config import Config


def load_search_space(path: str) -> dict[str, dict[str, Any]]:
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    return raw["search_space"]


def suggest_value(trial, key: str, spec: dict[str, Any]) -> Any:
    kind = spec["kind"]
    if kind == "categorical":
        return trial.suggest_categorical(key, spec["choices"])
    if kind == "int":
        return trial.suggest_int(key, spec["low"], spec["high"], step=spec.get("step", 1), log=spec.get("log", False))
    if kind == "float":
        return trial.suggest_float(key, spec["low"], spec["high"], log=spec.get("log", False))
    raise ValueError(f"unknown search-space kind: {kind!r} for key {key!r}")


def _set_nested(d: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor = d
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def build_search_space(trial, search_space_path: str, base_config_path: str) -> Config:
    space = load_search_space(search_space_path)
    with open(base_config_path) as fh:
        base_raw = yaml.safe_load(fh)

    for dotted_key, spec in space.items():
        value = suggest_value(trial, dotted_key, spec)
        _set_nested(base_raw, dotted_key, value)

    from dcnn_curvature.config import _dataclass_from_dict

    return _dataclass_from_dict(Config, base_raw)
