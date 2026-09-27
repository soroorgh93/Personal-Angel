"""Profile loading. Profiles are YAML files under config/ that may `_extends` one
another. Environment variables of the form ANGEL_<SECTION>__<KEY> override
values (e.g. ANGEL_LLM__BASE_URL=http://spark:8000/v1)."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"

def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result

def _load_yaml(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    seen = seen or set()
    path = path.resolve()
    if path in seen:
        raise ValueError(f"Circular _extends chain at {path}")
    seen.add(path)
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    parent_name = data.pop("_extends", None)
    if parent_name:
        parent = _load_yaml(path.parent / parent_name, seen)
        data = _deep_merge(parent, data)
    return data

def _apply_env_overrides(config: dict[str, Any]) -> None:
    prefix = "ANGEL_"
    for key, value in os.environ.items():
        if not key.startswith(prefix) or "__" not in key:
            continue
        section, _, name = key[len(prefix):].partition("__")
        section, name = section.lower(), name.lower()
        config.setdefault(section, {})
        current = config[section].get(name)
        if isinstance(current, bool):
            config[section][name] = value.lower() in {"1", "true", "yes", "on"}
        elif isinstance(current, int) and not isinstance(current, bool):
            config[section][name] = int(value)
        elif isinstance(current, float):
            config[section][name] = float(value)
        else:
            config[section][name] = value

def load_profile(name_or_path: str | Path = "fixture") -> dict[str, Any]:
    path = Path(name_or_path)
    if not path.suffix:
        path = CONFIG_DIR / f"{path}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Profile not found: {path}")
    config = _load_yaml(path)
    config["_profile_path"] = str(path)
    config["_project_root"] = str(PROJECT_ROOT)
    _apply_env_overrides(config)
    return config

def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(config.get("_project_root", PROJECT_ROOT)) / path
