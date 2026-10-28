"""Command-line entry point: python -m src.main --voice ... --text-file ... --output ..."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio_utils import AudioError, export_mp3, validate_reference
from .pipeline import assemble, generate_chunks
from .text_utils import TextError, load_pronunciations, normalize_text, split_into_chunks
from .voice_engine import MODEL_KINDS, VoiceEngineError, load_model, pick_device


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Generate narration in a cloned voice.")
    p.add_argument("--voice", required=True, help="Reference voice (.wav or .flac)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="Script text")
    src.add_argument("--text-file", help="Path to a UTF-8 text file with the script")
    p.add_argument("--output", default="output/narration.wav", help="Output WAV path")
    p.add_argument("--model", choices=MODEL_KINDS, default="turbo")
    p.add_argument("--seed", type=int, default=1, help="Base random seed (default 1)")
    p.add_argument("--pronunciations", default="pronunciations.json")
    p.add_argument("--max-chars", type=int, default=250, help="Split sentences longer than this")
    p.add_argument("--sentence-pause", type=float, default=0.35, help="Seconds between sentences")
    p.add_argument("--paragraph-pause", type=float, default=0.8, help="Seconds between paragraphs")
    p.add_argument("--work-dir", default=None, help="Chunk cache folder (default output/work/<name>)")
    p.add_argument("--regen", default="", help="Sentence numbers to regenerate, e.g. 3,7")
    p.add_argument("--force", action="store_true", help="Regenerate every sentence")
    p.add_argument("--mp3", action="store_true", help="Also export an MP3 next to the WAV")
    return p.parse_args(argv)


def _parse_regen(value: str, total: int) -> set[int]:
    if not value.strip():
        return set()
    try:
        numbers = {int(x) for x in value.split(",") if x.strip()}
    except ValueError:
        raise VoiceEngineError(f"--regen must be comma-separated numbers, got '{value}'")
    bad = sorted(n for n in numbers if not 1 <= n <= total)
    if bad:
        raise VoiceEngineError(f"--regen numbers out of range 1..{total}: {bad}")
    return numbers


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        report = validate_reference(args.voice)
        print(f"Reference: {report.seconds:.1f}s, {report.sample_rate} Hz, "
              f"{report.channels} ch, peak {report.peak:.2f}, RMS {report.rms_db:.0f} dBFS")
        for w in report.warnings:
            print(f"  warning: {w}")

        if args.text_file:
            text_path = Path(args.text_file)
            if not text_path.is_file():
                raise VoiceEngineError(f"Text file not found: {text_path}")
            raw = text_path.read_text(encoding="utf-8")
        else:
            raw = args.text

        text = normalize_text(raw, load_pronunciations(args.pronunciations))
        chunks = split_into_chunks(text, args.max_chars)
        if not chunks:
            raise VoiceEngineError("The script is empty.")
        regen = _parse_regen(args.regen, len(chunks))

        print(f"\n{len(chunks)} sentences:")
        for n, c in enumerate(chunks, start=1):
            print(f"{n:>4}. {c.text[:80]}{'...' if len(c.text) > 80 else ''}")
        print()

        output = Path(args.output)
        work_dir = Path(args.work_dir) if args.work_dir else Path("output/work") / output.stem

        models = {}

        def get_model():
            if "m" not in models:
                device = pick_device()
                print(f"Loading {args.model} model on {device}...")
                t0 = time.time()
                models["m"] = load_model(args.model, device)
                print(f"Model ready in {time.time() - t0:.1f}s")
            return models["m"]

        t1 = time.time()
        paths = generate_chunks(get_model, chunks, args.voice, args.model, work_dir,
                                args.seed, regen, args.force)
        audio, sr = assemble(paths, chunks, args.sentence_pause, args.paragraph_pause)

        output.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output), np.clip(audio, -1.0, 1.0), sr, subtype="PCM_16")
        print(f"\nSaved {output} ({len(audio) / sr:.1f}s of audio, total {time.time() - t1:.1f}s)")

        if args.mp3:
            mp3 = export_mp3(output, output.with_suffix(".mp3"))
            print(f"Saved {mp3}")
        return 0
    except (VoiceEngineError, AudioError, TextError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
