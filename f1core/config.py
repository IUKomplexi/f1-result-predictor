"""Configuration loader + writer (stdlib ``tomllib``/``os.replace``; no third-party dependency).

Reads ``config.toml`` at the repo root and merges it over built-in defaults,
so every CLI works out of the box even without a config file present. The file
is the single source of truth: :func:`save_config` writes it back in place
(atomically) so the CLI and the web dashboard always read the same settings.
"""

from __future__ import annotations

import copy
import json
import os
import tomllib
from pathlib import Path
from typing import Any

DEFAULTS: dict[str, dict[str, Any]] = {
    "api": {
        "base_url": "https://api.jolpi.ca/ergast/f1",
        "user_agent": (
            "f1-result-predictor/0.1.0 "
            "(https://github.com/example/f1-result-predictor; contact: dev@example.com)"
        ),
        "sleep_seconds": 0.25,
        "timeout": 30.0,
        "max_retries": 3,
    },
    "data": {
        "cache_dir": "data/raw",
        "dataset": "data/features.parquet",
        "start_season": 2010,
        "end_season": 2026,
    },
    "model": {
        "checkpoint": "data/model/hurdle.joblib",
        "calibrators": "data/model/calibrators.joblib",
        "seed": 42,
        # HGB hyperparameters (see model/search.py to tune). Train reads these
        # from config first; model/train.DEFAULT_PARAMS is the code fallback.
        "params": {
            "max_iter": 400,
            "learning_rate": 0.03,
            "max_depth": 3,
            "l2_regularization": 1.0,
            "min_samples_leaf": 20,
        },
    },
    "report": {
        "backtest": "reports/backtest.md",
        "prediction": "reports/prediction.md",
    },
    "features": {
        # None ⇒ registry defaults (features/registry.py): core on,
        # selectable/cut off. An explicit list overrides the defaults.
        "enabled": None,
    },
    "weather": {
        "cache_dir": "data/weather",
    },
}

# Season bounds for validation / the UI season pickers.
SEASON_MIN = 1950
SEASON_MAX = 2100

# Modern-era floor for the UI season pickers (2014 hybrid era onwards).
# Config validation stays permissive (SEASON_MIN) so legacy configs keep
# loading; the dashboard clamps pipeline runs to this floor.
DATA_START_FLOOR = 2014

# Model hyperparameter keys accepted under [model.params] (HGB constructor
# args). Unknown keys are rejected so a typo can't silently change behaviour.
MODEL_PARAM_KEYS = {"max_iter", "learning_rate", "max_depth",
                    "l2_regularization", "min_samples_leaf"}


def load_config(path: str | Path = "config.toml") -> dict[str, dict[str, Any]]:
    """Load ``config.toml`` merged over :data:`DEFAULTS` (section-wise)."""
    cfg = copy.deepcopy(DEFAULTS)
    p = Path(path)
    if not p.exists():
        return cfg
    with p.open("rb") as fh:
        user = tomllib.load(fh)
    for section, values in user.items():
        if section in cfg and isinstance(values, dict):
            cfg[section].update(values)
        else:
            cfg[section] = values
    return cfg


# --------------------------------------------------------------------------
# Schema + validation (used by GET/PUT /api/config and save_config)
# --------------------------------------------------------------------------

# JSON-serializable field descriptors for the UI form generator. ``type`` is
# one of str / int / float / bool / list[str] / params / features.
SCHEMA: list[dict[str, Any]] = [
    {"section": "api", "key": "base_url", "type": "str",
     "help": "Jolpica (Ergast-compatible) API base URL"},
    {"section": "api", "key": "user_agent", "type": "str",
     "help": "HTTP User-Agent (Jolpica terms require a descriptive one)"},
    {"section": "api", "key": "sleep_seconds", "type": "float",
     "min": 0.0, "max": 60.0, "help": "Seconds between API requests"},
    {"section": "api", "key": "timeout", "type": "float",
     "min": 1.0, "max": 300.0, "help": "Request timeout (seconds)"},
    {"section": "api", "key": "max_retries", "type": "int",
     "min": 0, "max": 20, "help": "Request retries before failing"},

    {"section": "data", "key": "cache_dir", "type": "str",
     "help": "Directory for cached raw API responses"},
    {"section": "data", "key": "dataset", "type": "str",
     "help": "Path to the featured parquet dataset"},
    {"section": "data", "key": "start_season", "type": "int",
     "min": SEASON_MIN, "max": SEASON_MAX,
     "help": "First training season. The dashboard clamps pipeline runs to "
             "the modern era (2014+)."},
    {"section": "data", "key": "end_season", "type": "int",
     "min": SEASON_MIN, "max": SEASON_MAX,
     "help": "Last training season. The live data ends here; the dashboard "
             "clamps pipeline runs to the latest season with fetched data."},

    {"section": "model", "key": "checkpoint", "type": "str",
     "help": "Model checkpoint path"},
    {"section": "model", "key": "calibrators", "type": "str",
     "help": "Calibrator checkpoint path"},
    {"section": "model", "key": "seed", "type": "int",
     "min": 0, "max": 2**31 - 1, "help": "Random seed for model fitting"},
    {"section": "model", "key": "params", "type": "params",
     "help": "HGB hyperparameters (max_iter, learning_rate, max_depth, "
             "l2_regularization, min_samples_leaf)"},

    {"section": "report", "key": "backtest", "type": "str",
     "help": "Backtest markdown report path"},
    {"section": "report", "key": "prediction", "type": "str",
     "help": "Prediction markdown report path"},

    {"section": "features", "key": "enabled", "type": "features",
     "help": "Explicit feature selection; empty falls back to registry defaults"},

    {"section": "weather", "key": "cache_dir", "type": "str",
     "help": "Directory for cached weather data"},
]


def _as_number(value: Any, kind: type) -> tuple[Any, str | None]:
    """Coerce ``value`` to int/float (``kind`` is ``int`` or ``float``)."""
    try:
        return kind(value), None
    except (TypeError, ValueError) as exc:
        return None, f"expected a number: {exc}"


def validate_config(cfg: dict) -> list[str]:
    """Validate a full config dict; returns a list of error strings (empty = ok).

    Checks types, numeric ranges, season ordering, model params, and feature
    names (against ``features/registry.py``).
    """
    errors: list[str] = []

    def val(section: str, key: str, value: Any) -> None:
        desc = next((d for d in SCHEMA if d["section"] == section and d["key"] == key), None)
        if desc is None:
            return
        kind = desc["type"]
        if kind == "str":
            if not isinstance(value, str):
                errors.append(f"{section}.{key}: expected a string")
            return
        if kind in ("int", "float"):
            if isinstance(value, bool):
                errors.append(f"{section}.{key}: expected a number, not a boolean")
                return
            if kind == "int":
                try:
                    fval = float(value)
                except (TypeError, ValueError):
                    errors.append(f"{section}.{key}: expected an integer")
                    return
                if not fval.is_integer():
                    errors.append(f"{section}.{key}: expected an integer")
                    return
                num = int(fval)
            else:
                num, err = _as_number(value, float)
                if err:
                    errors.append(f"{section}.{key}: {err}")
                    return
            if "min" in desc and num < desc["min"]:
                errors.append(f"{section}.{key}: must be >= {desc['min']}")
            if "max" in desc and num > desc["max"]:
                errors.append(f"{section}.{key}: must be <= {desc['max']}")
            return
        if kind == "params":
            if not isinstance(value, dict):
                errors.append("model.params: expected a table")
                return
            for pkey, pvalue in value.items():
                if pkey not in MODEL_PARAM_KEYS:
                    errors.append(f"model.params: unknown parameter {pkey!r}")
                elif not isinstance(pvalue, (int, float)):
                    errors.append(f"model.params.{pkey}: expected a number")
            return
        if kind == "features":
            if value is not None and not isinstance(value, list):
                errors.append("features.enabled: expected a list or null")
            return
        if kind == "list[str]":
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                errors.append(f"{section}.{key}: expected a list of strings")

    for section, values in cfg.items():
        if not isinstance(values, dict):
            errors.append(f"{section}: expected a table")
            continue
        for key, value in values.items():
            val(section, key, value)

    # Season ordering (when present and numeric).
    data = cfg.get("data", {})
    start, end = data.get("start_season"), data.get("end_season")
    if isinstance(start, (int, float)) and isinstance(end, (int, float)) and start > end:
        errors.append("data: start_season must be <= end_season")

    # Feature names against the registry.
    enabled = (cfg.get("features") or {}).get("enabled")
    if enabled:
        from features.registry import all_feature_ids  # local: avoid hard import cycle

        known = set(all_feature_ids())
        unknown = [f for f in enabled if f not in known]
        if unknown:
            errors.append(f"features.enabled: unknown feature(s): {unknown}")

    return errors


# --------------------------------------------------------------------------
# TOML writer (hand-rolled, stdlib-only: values are scalars / lists / dicts)
# --------------------------------------------------------------------------

def _format_value(value: Any) -> str | None:
    """TOML literal for a scalar/list, or None for ``None`` (written as a comment)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)  # a valid TOML basic string
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(v for v in (_format_value(i) for i in value) if v is not None) + "]"
    raise TypeError(f"cannot serialize {value!r} to TOML")


_SECTION_ORDER = ["api", "data", "model", "report", "features", "weather"]


def config_to_toml(cfg: dict) -> str:
    """Serialize an effective config dict back to TOML, preserving section order.

    ``None`` values become comments (the ``[features] enabled`` convention);
    nested dicts (``[model.params]``) are written as sub-tables.
    """
    lines = [
        "# f1-result-predictor configuration.",
        "# Every value here is optional: the built-in defaults in config.py already",
        "# match this file. CLI flags override config values.",
    ]
    for section in _SECTION_ORDER:
        values = cfg.get(section, {})
        if not isinstance(values, dict):
            continue
        flat = {k: v for k, v in values.items() if not isinstance(v, dict)}
        subs = {k: v for k, v in values.items() if isinstance(v, dict)}
        lines.append(f"\n[{section}]")
        for key, value in flat.items():
            formatted = _format_value(value)
            if formatted is None:
                if key == "enabled":
                    lines.append(
                        "# enabled = <registry defaults> (core features on, "
                        "selectable/cut off). Edit the list to override."
                    )
                else:
                    lines.append(f"# {key} = <unset>")
            else:
                lines.append(f"{key} = {formatted}")
        for sub, subvalues in subs.items():
            lines.append(f"\n[{section}.{sub}]")
            for key, value in subvalues.items():
                formatted = _format_value(value)
                lines.append(f"{key} = {formatted}" if formatted is not None
                             else f"# {key} = <unset>")
    return "\n".join(lines) + "\n"


def save_config(cfg: dict, path: str | Path = "config.toml") -> Path:
    """Validate and atomically write ``cfg`` to ``path``. Returns the path.

    Raises :class:`ValueError` with the first validation error if invalid.
    """
    errors = validate_config(cfg)
    if errors:
        raise ValueError(errors[0])
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = config_to_toml(cfg)
    # Atomic write: write to a sibling temp file, fsync, then replace.
    tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, p)
    return p
