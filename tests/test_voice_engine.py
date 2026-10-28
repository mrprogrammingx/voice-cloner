import sys
import types

import pytest
import torch

from src.voice_engine import VoiceEngineError, load_model, save_wav, synthesize


class RandomModel:
    sr = 24000

    def generate(self, text, **kwargs):
        return torch.rand(1, 1000)


class OutOfMemoryModel:
    sr = 24000

    def generate(self, text, **kwargs):
        raise RuntimeError("MPS backend out of memory")


def test_empty_script_rejected(fake_model, voice_wav):
    with pytest.raises(VoiceEngineError, match="empty"):
        synthesize(fake_model, "   ", voice_wav)


def test_missing_voice_rejected(fake_model, tmp_path):
    with pytest.raises(VoiceEngineError, match="not found"):
        synthesize(fake_model, "Hello.", tmp_path / "nope.wav")


def test_unknown_model_kind():
    with pytest.raises(VoiceEngineError, match="Unknown model"):
        load_model("does-not-exist")


def test_model_load_failure_is_reported(monkeypatch):
    class Boom:
        @staticmethod
        def from_pretrained(device):
            raise RuntimeError("weights missing")

    fake_module = types.ModuleType("chatterbox.tts_turbo")
    fake_module.ChatterboxTurboTTS = Boom
    monkeypatch.setitem(sys.modules, "chatterbox", types.ModuleType("chatterbox"))
    monkeypatch.setitem(sys.modules, "chatterbox.tts_turbo", fake_module)
    with pytest.raises(VoiceEngineError, match="weights missing"):
        load_model("turbo", "cpu")


def test_generation_failure_is_wrapped(voice_wav):
    with pytest.raises(VoiceEngineError, match="out of memory"):
        synthesize(OutOfMemoryModel(), "Hello.", voice_wav)


def test_same_seed_gives_same_audio(voice_wav):
    a = synthesize(RandomModel(), "Hello.", voice_wav, seed=5)
    b = synthesize(RandomModel(), "Hello.", voice_wav, seed=5)
    c = synthesize(RandomModel(), "Hello.", voice_wav, seed=6)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_save_wav_creates_folders_and_file(tmp_path, fake_model, voice_wav):
    wav = synthesize(fake_model, "Hello.", voice_wav)
    out = save_wav(wav, fake_model.sr, tmp_path / "deep" / "dir" / "out.wav")
    assert out.is_file() and out.stat().st_size > 0
