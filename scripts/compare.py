"""Compare takes: python scripts/compare.py --model original|turbo --voice samples/ref_480.wav --takes 3"""
import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torchaudio as ta

p = argparse.ArgumentParser()
p.add_argument("--model", choices=["original", "turbo"], required=True)
p.add_argument("--voice", required=True)
p.add_argument("--takes", type=int, default=3)
args = p.parse_args()

device = "mps" if torch.backends.mps.is_available() else "cpu"
original_load = torch.load
torch.load = lambda *a, **k: original_load(*a, **{**k, "map_location": k.get("map_location", torch.device(device))})

if args.model == "turbo":
    from chatterbox.tts_turbo import ChatterboxTurboTTS as Model
else:
    from chatterbox.tts import ChatterboxTTS as Model

TEXT = ("In this tutorial, we will build a Python API with async functions. "
        "First, we connect to a PostgreSQL database and run a simple SQL query. "
        "Then we return the results as JSON.")

model = Model.from_pretrained(device=device)
out_dir = Path("output/compare")
out_dir.mkdir(parents=True, exist_ok=True)
tag = Path(args.voice).stem

for seed in range(1, args.takes + 1):
    torch.manual_seed(seed)
    t = time.perf_counter()
    wav = model.generate(TEXT, audio_prompt_path=args.voice)
    dt = time.perf_counter() - t
    path = out_dir / f"{args.model}_{tag}_seed{seed}.wav"
    ta.save(str(path), wav, model.sr)
    print(f"[take {seed}] {wav.shape[-1] / model.sr:.1f}s audio in {dt:.1f}s -> {path}")
