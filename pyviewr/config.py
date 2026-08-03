"""Persist simple app settings (save directory)."""

from __future__ import annotations

import json
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
DEFAULT_SAVE_DIR = Path.home() / "Pictures" / "Pyviewr"


def load_save_dir() -> Path:
    if CONFIG_PATH.is_file():
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            raw = data.get("save_dir") or ""
            if raw:
                return Path(raw).expanduser()
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    return DEFAULT_SAVE_DIR


def save_save_dir(path: Path | str) -> Path:
    resolved = Path(path).expanduser()
    resolved.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps({"save_dir": str(resolved)}, indent=2) + "\n",
        encoding="utf-8",
    )
    return resolved
