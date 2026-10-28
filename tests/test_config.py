import json

import pytest

from src.config import DEFAULTS, ConfigError, load_config, save_config


def test_defaults_when_file_missing(tmp_path):
    assert load_config(tmp_path / "none.json") == DEFAULTS


def test_file_overrides_defaults(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"seed": 7, "mp3": False}))
    cfg = load_config(p)
    assert cfg["seed"] == 7 and cfg["mp3"] is False and cfg["model"] == DEFAULTS["model"]


def test_unknown_setting_rejected(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"sed": 1}))
    with pytest.raises(ConfigError, match="Unknown setting"):
        load_config(p)


def test_wrong_type_rejected(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"seed": "one"}))
    with pytest.raises(ConfigError, match="wrong type"):
        load_config(p)


def test_invalid_json_rejected(tmp_path):
    p = tmp_path / "c.json"
    p.write_text("{not json")
    with pytest.raises(ConfigError, match="Invalid JSON"):
        load_config(p)


def test_integer_pause_is_accepted(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"sentence_pause": 1}))
    assert load_config(p)["sentence_pause"] == 1


def test_save_and_load_roundtrip(tmp_path):
    p = tmp_path / "c.json"
    cfg = dict(DEFAULTS, voice="samples/other.wav")
    save_config(cfg, p)
    assert load_config(p) == cfg
