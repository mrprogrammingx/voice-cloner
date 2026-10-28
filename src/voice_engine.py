"""Thin wrapper around Chatterbox for local voice cloning (Turbo by default)."""
import os
from pathlib import Path

# Let PyTorch fall back to CPU for any op MPS doesn't support yet.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torchaudio as ta

MODEL_KINDS = ("turbo", "original")


class VoiceEngineError(Exception):
    """Raised for any user-facing failure (bad input, model won't load, etc.)."""


def pick_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(kind: str = "turbo", device: str | None = None):
    """Load a pretrained Chatterbox model (downloads weights on first run)."""
    if kind not in MODEL_KINDS:
        raise VoiceEngineError(f"Unknown model '{kind}'. Choose from: {', '.join(MODEL_KINDS)}")
    device = device or pick_device()

    # Same workaround as the official example_for_mac.py: map saved weights
    # onto the chosen device instead of the one they were saved from.
    original_load = torch.load

    def patched_load(*args, **kwargs):
        kwargs.setdefault("map_location", torch.device(device))
        return original_load(*args, **kwargs)

    torch.load = patched_load
    try:
        if kind == "turbo":
            from chatterbox.tts_turbo import ChatterboxTurboTTS as Model
        else:
            from chatterbox.tts import ChatterboxTTS as Model
        return Model.from_pretrained(device=device)
    except Exception as exc:
        raise VoiceEngineError(f"Could not load the '{kind}' model on '{device}': {exc}") from exc
    finally:
        torch.load = original_load


def synthesize(model, text: str, voice_path: str | Path, kind: str = "turbo",
               seed: int | None = None, exaggeration: float = 0.5, cfg_weight: float = 0.5):
    """Generate speech for `text` in the voice of the reference clip.

    exaggeration / cfg_weight apply to the original model only.
    Returns a tensor shaped (1, samples) at model.sr.
    """
    text = text.strip()
    if not text:
        raise VoiceEngineError("The script is empty.")
    voice_path = Path(voice_path)
    if not voice_path.is_file():
        raise VoiceEngineError(f"Reference voice not found: {voice_path}")
    if seed is not None:
        torch.manual_seed(seed)
    kwargs = {"audio_prompt_path": str(voice_path)}
    if kind == "original":
        kwargs.update(exaggeration=exaggeration, cfg_weight=cfg_weight)
    try:
        return model.generate(text, **kwargs)
    except Exception as exc:
        raise VoiceEngineError(f"Generation failed: {exc}") from exc


def save_wav(wav, sample_rate: int, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ta.save(str(output_path), wav, sample_rate)
    return output_path
