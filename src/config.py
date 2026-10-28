"""Reusable settings in config.json (command-line options override them)."""
import json
from pathlib import Path

CONFIG_PATH = Path("config.json")

DEFAULTS = {
    "voice": "samples/ref_480.wav",
    "model": "vc",
    "seed": 1,
    "max_chars": 250,
    "sentence_pause": 0.35,
    "paragraph_pause": 0.8,
    "pronunciations": "pronunciations.json",
    "output_dir": "output",
    "mp3": True,
}


class ConfigError(Exception):
    """Raised when config.json is unreadable or has bad settings."""


def _type_ok(value, default) -> bool:
    if isinstance(default, bool):
        return isinstance(value, bool)
    if isinstance(default, float):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if isinstance(default, int):
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, type(default))


def load_config(path: str | Path = CONFIG_PATH) -> dict:
    config = dict(DEFAULTS)
    path = Path(path)
    if not path.is_file():
        return config
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a JSON object.")
    unknown = sorted(set(data) - set(DEFAULTS))
    if unknown:
        raise ConfigError(f"Unknown setting(s) in {path}: {', '.join(unknown)}")
    for key, value in data.items():
        if not _type_ok(value, DEFAULTS[key]):
            raise ConfigError(f"Setting '{key}' in {path} has the wrong type: {value!r}")
    config.update(data)
    return config


def save_config(config: dict, path: str | Path = CONFIG_PATH) -> None:
    Path(path).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
