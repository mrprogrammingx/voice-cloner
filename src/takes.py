"""Audition several takes of one sentence, then pick the best.

  python -m src.takes --voice V --text-file F --output O --sentence 3 --takes 4
  python -m src.takes --voice V --text-file F --output O --sentence 3 --pick 2

Use the same --voice/--text-file/--output/--seed/--model as your main.py run so
the sentence numbers and cache folder match.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import soundfile as sf

from .audio_utils import AudioError, validate_reference
from .text_utils import TextError, load_pronunciations, normalize_text, split_into_chunks
from .voice_engine import MODEL_KINDS, VoiceEngineError, load_model, pick_device, synthesize


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Audition takes of one sentence.")
    p.add_argument("--voice", required=True)
    p.add_argument("--text-file", required=True)
    p.add_argument("--output", default="output/narration.wav", help="Same as the main.py run (names the cache)")
    p.add_argument("--sentence", type=int, required=True, help="Sentence number from the main.py list")
    p.add_argument("--takes", type=int, default=4, help="How many takes to generate")
    p.add_argument("--pick", type=int, default=None, help="Take number to keep (skips generation)")
    p.add_argument("--model", choices=MODEL_KINDS, default="turbo")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--pronunciations", default="pronunciations.json")
    p.add_argument("--max-chars", type=int, default=250)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        validate_reference(args.voice)
        text_path = Path(args.text_file)
        if not text_path.is_file():
            raise VoiceEngineError(f"Text file not found: {text_path}")
        text = normalize_text(text_path.read_text(encoding="utf-8"), load_pronunciations(args.pronunciations))
        chunks = split_into_chunks(text, args.max_chars)
        n = args.sentence
        if not 1 <= n <= len(chunks):
            raise VoiceEngineError(f"--sentence must be 1..{len(chunks)}")

        work_dir = Path("output/work") / Path(args.output).stem
        takes_dir = work_dir / "takes"
        chunk_path = work_dir / f"chunk_{n:04d}.wav"

        if args.pick is not None:
            take = takes_dir / f"chunk_{n:04d}_take{args.pick}.wav"
            if not take.is_file():
                raise VoiceEngineError(f"Take not found: {take}. Generate takes first.")
            work_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(take, chunk_path)
            manifest_path = work_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
            settings = {"voice": str(args.voice), "model": args.model, "seed": args.seed}
            if any(manifest.get(k) != v for k, v in settings.items()):
                manifest = {**settings, "chunks": []}
            entries = manifest.get("chunks", [])
            entries += [{"text": "", "paragraph_end": False, "attempt": 0} for _ in range(len(chunks) - len(entries))]
            entries[n - 1] = {"text": chunks[n - 1].text, "paragraph_end": chunks[n - 1].paragraph_end, "attempt": 0}
            manifest["chunks"] = entries
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"Sentence {n} now uses take {args.pick}. Re-run main.py to rebuild the narration.")
            return 0

        print(f"Sentence {n}: {chunks[n - 1].text}\n")
        takes_dir.mkdir(parents=True, exist_ok=True)
        print(f"Loading {args.model} model...")
        model = load_model(args.model, pick_device())
        for k in range(1, args.takes + 1):
            seed = args.seed + n + 100000 * k
            start = time.time()
            wav = synthesize(model, chunks[n - 1].text, args.voice, args.model, seed)
            path = takes_dir / f"chunk_{n:04d}_take{k}.wav"
            sf.write(str(path), wav.squeeze().detach().cpu().numpy().astype("float32"), model.sr, subtype="FLOAT")
            print(f"take {k}: {len(wav.squeeze()) / model.sr:.1f}s in {time.time() - start:.1f}s -> {path}")
        print(f"\nListen with: afplay {takes_dir}/chunk_{n:04d}_take<K>.wav")
        print(f"Keep one with: python -m src.takes ... --sentence {n} --pick <K>")
        return 0
    except (VoiceEngineError, AudioError, TextError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
