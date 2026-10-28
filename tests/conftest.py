import math

import numpy as np
import pytest
import soundfile as sf
import torch


class FakeModel:
    """Stands in for a Chatterbox model: returns a short sine tone, records calls."""
    sr = 24000

    def __init__(self):
        self.calls = []

    def generate(self, text, **kwargs):
        self.calls.append(text)
        t = torch.arange(int(self.sr * 0.5)) / self.sr
        return (0.3 * torch.sin(2 * math.pi * 220 * t)).unsqueeze(0)


@pytest.fixture
def fake_model():
    return FakeModel()


@pytest.fixture
def voice_wav(tmp_path):
    sr = 24000
    t = np.arange(sr * 10) / sr
    path = tmp_path / "voice.wav"
    sf.write(str(path), (0.2 * np.sin(2 * np.pi * 220 * t)).astype("float32"), sr)
    return path
