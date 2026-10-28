"""Does the reference clip carry accent/pronunciation habits? Compare default voice vs ref."""
import gc
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torchaudio as ta

TEXT = ("The function will return a value. We await the async call, then deploy the app. "
        "This lesson covers data and Python.")
REF = "samples/ref_480.wav"
out = Path("output/accent_test")
out.mkdir(parents=True, exist_ok=True)

device = "mps" if torch.backends.mps.is_available() else "cpu"
original_load = torch.load
torch.load = lambda *a, **k: original_load(*a, **{**k, "map_location": k.get("map_location", torch.device(device))})

from chatterbox.tts import ChatterboxTTS
model = ChatterboxTTS.from_pretrained(device=device)
for seed in (1, 2):
    torch.manual_seed(seed)
    wav = model.generate(TEXT)  # no audio prompt: built-in default voice
    path = out / f"default_voice_original_seed{seed}.wav"
    ta.save(str(path), wav, model.sr)
    print("saved", path)
del model
gc.collect()
if device == "mps":
    torch.mps.empty_cache()

from chatterbox.tts_turbo import ChatterboxTurboTTS
model = ChatterboxTurboTTS.from_pretrained(device=device)
for seed in (1, 2, 3):
    torch.manual_seed(seed)
    wav = model.generate(TEXT, audio_prompt_path=REF)
    path = out / f"your_ref_turbo_seed{seed}.wav"
    ta.save(str(path), wav, model.sr)
    print("saved", path)
