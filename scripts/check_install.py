"""Phase 3 check: confirm torch/MPS work and the Chatterbox model loads."""
import os
import time

# Lets PyTorch fall back to CPU for any op MPS doesn't support yet.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

print(f"torch {torch.__version__}")
print(f"MPS built: {torch.backends.mps.is_built()}  available: {torch.backends.mps.is_available()}")

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Same workaround as the official example_for_mac.py: map saved weights to our device.
_original_load = torch.load


def _patched_load(*args, **kwargs):
    kwargs.setdefault("map_location", torch.device(device))
    return _original_load(*args, **kwargs)


torch.load = _patched_load

from chatterbox.tts import ChatterboxTTS

start = time.time()
model = ChatterboxTTS.from_pretrained(device=device)
print(f"Model loaded in {time.time() - start:.1f}s, sample rate: {model.sr}")
