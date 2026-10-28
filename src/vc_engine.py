"""Two-stage engine: default-voice speech (original Chatterbox) -> voice conversion to your voice."""
import tempfile
from pathlib import Path

import torchaudio as ta


class VCBundle:
    """Looks like a Chatterbox model: has .sr and .generate(text, audio_prompt_path=...)."""

    def __init__(self, tts, vc):
        self.tts = tts
        self.vc = vc
        self.sr = vc.sr

    @classmethod
    def load(cls, device: str):
        from chatterbox.tts import ChatterboxTTS
        from chatterbox.vc import ChatterboxVC

        return cls(ChatterboxTTS.from_pretrained(device=device),
                   ChatterboxVC.from_pretrained(device=device))

    def generate(self, text: str, audio_prompt_path):
        source = self.tts.generate(text)  # built-in default voice: clean pronunciation
        with tempfile.TemporaryDirectory() as tmp:
            src_path = Path(tmp) / "source.wav"
            ta.save(str(src_path), source, self.tts.sr)
            return self.vc.generate(str(src_path), target_voice_path=str(audio_prompt_path))
