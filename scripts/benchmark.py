"""Speed test: python scripts/benchmark.py --model original|turbo --device mps|cpu"""
import argparse
import os
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

p = argparse.ArgumentParser()
p.add_argument("--model", choices=["original", "turbo"], required=True)
p.add_argument("--device", choices=["mps", "cpu"], required=True)
p.add_argument("--voice", default="samples/my_voice.wav")
args = p.parse_args()

original_load = torch.load
torch.load = lambda *a, **k: original_load(*a, **{**k, "map_location": k.get("map_location", torch.device(args.device))})

if args.model == "turbo":
    from chatterbox.tts_turbo import ChatterboxTurboTTS as Model
else:
    from chatterbox.tts import ChatterboxTTS as Model

t0 = time.perf_counter()
model = Model.from_pretrained(device=args.device)
print(f"[load] {time.perf_counter() - t0:.1f}s")

text = "In this tutorial, we will build a Python API with async functions."
for run in (1, 2):
    t = time.perf_counter()
    wav = model.generate(text, audio_prompt_path=args.voice)
    if args.device == "mps":
        torch.mps.synchronize()
    dt = time.perf_counter() - t
    secs = wav.shape[-1] / model.sr
    print(f"[run {run}] {secs:.1f}s audio in {dt:.1f}s  ->  {dt / secs:.1f}x slower than real time")

import torchaudio as ta
ta.save(f"output/bench_{args.model}_{args.device}.wav", wav, model.sr)
print(f"[saved] output/bench_{args.model}_{args.device}.wav")
