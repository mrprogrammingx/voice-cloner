"""Voice-convert the default-voice samples into the ref_480 voice."""
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torchaudio as ta

device = "mps" if torch.backends.mps.is_available() else "cpu"
original_load = torch.load
torch.load = lambda *a, **k: original_load(*a, **{**k, "map_location": k.get("map_location", torch.device(device))})

from chatterbox.vc import ChatterboxVC

model = ChatterboxVC.from_pretrained(device=device)
src_dir = Path("output/accent_test")
for src in sorted(src_dir.glob("default_voice_original_seed*.wav")):
    wav = model.generate(str(src), target_voice_path="samples/ref_480.wav")
    out = src_dir / src.name.replace("default_voice_original", "vc_to_your_voice")
    ta.save(str(out), wav, model.sr)
    print("saved", out)
