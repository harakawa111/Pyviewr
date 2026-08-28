"""Persist app settings in config.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
DEFAULT_SAVE_DIR = Path.home() / "Pictures" / "Pyviewr"

_CAMERA_KEYS = ("ExposureTime", "Gain", "Gamma")


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.is_file():
        return {}
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_config(data: dict[str, Any]) -> None:
    CONFIG_PATH.write_text(
        json.dumps(data, indent=2) + "\n",
        encoding="utf-8",
    )


def update_config(**patches: Any) -> dict[str, Any]:
    """Shallow-merge top-level keys and write config.json."""
    data = load_config()
    for key, value in patches.items():
        if value is None:
            data.pop(key, None)
        elif isinstance(value, dict) and isinstance(data.get(key), dict):
            merged = dict(data[key])
            merged.update(value)
            data[key] = merged
        else:
            data[key] = value
    save_config(data)
    return data


def load_save_dir() -> Path:
    raw = load_config().get("save_dir") or ""
    if raw:
        return Path(str(raw)).expanduser()
    return DEFAULT_SAVE_DIR


def save_save_dir(path: Path | str) -> Path:
    resolved = Path(path).expanduser()
    resolved.mkdir(parents=True, exist_ok=True)
    update_config(save_dir=str(resolved))
    return resolved


def load_camera_features() -> dict[str, float]:
    raw = load_config().get("camera")
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for key in _CAMERA_KEYS:
        if key in raw:
            try:
                out[key] = float(raw[key])
            except (TypeError, ValueError):
                pass
    return out


def load_timer_seconds() -> int:
    raw = load_config().get("timer_seconds", 0)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 0
    return value if value > 0 else 0
