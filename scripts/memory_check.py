"""Repeat real generations and print memory use: python scripts/memory_check.py --model vc --rounds 4"""
import argparse
import resource
import subprocess
import time

from src.voice_engine import load_model, pick_device, synthesize

p = argparse.ArgumentParser()
p.add_argument("--model", default="vc")
p.add_argument("--voice", default="samples/ref_480.wav")
p.add_argument("--rounds", type=int, default=4)
args = p.parse_args()

import torch

device = pick_device()
model = load_model(args.model, device)
text = "The function will return a value, and we await the async call."
for i in range(1, args.rounds + 1):
    t = time.time()
    synthesize(model, text, args.voice, args.model, seed=i)
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024  # bytes on macOS
    mps = ""
    if device == "mps":
        mps = f", mps allocated {torch.mps.current_allocated_memory() / 1e6:.0f} MB"
    swap = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.strip()
    print(f"round {i}: {time.time() - t:.1f}s, peak RSS {rss_mb:.0f} MB{mps}\n         swap: {swap}")
