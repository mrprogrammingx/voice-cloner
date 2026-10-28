"""Command line: python -m src.main --text-file scripts/tutorial.txt  (defaults come from config.json)."""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio_utils import AudioError, export_mp3, validate_reference
from .config import ConfigError, load_config, save_config
from .genlog import log_event
from .pipeline import assemble, generate_chunks
from .text_utils import TextError, load_pronunciations, normalize_text, split_into_chunks
from .voice_engine import MODEL_KINDS, VoiceEngineError, load_model, pick_device

LOG_NAME = "generation_log.jsonl"


def parse_args(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="config.json", help="Settings file (default config.json)")
    known, _ = pre.parse_known_args(argv)
    config = load_config(known.config)

    p = argparse.ArgumentParser(description="Generate narration in a cloned voice.", parents=[pre])
    p.add_argument("--voice", default=config["voice"], help="Reference voice (.wav or .flac)")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--text", help="Script text")
    src.add_argument("--text-file", nargs="+", help="One or more UTF-8 script files")
    src.add_argument("--batch-dir", help="Folder of .txt scripts to narrate")
    p.add_argument("--output", help="Output WAV path (single script only; default: automatic name)")
    p.add_argument("--output-dir", default=config["output_dir"], help="Where automatic outputs go")
    p.add_argument("--work-dir", help="Chunk cache folder (default <output-dir>/work/<script name>)")
    p.add_argument("--model", choices=MODEL_KINDS, default=config["model"])
    p.add_argument("--seed", type=int, default=config["seed"], help="Base random seed")
    p.add_argument("--pronunciations", default=config["pronunciations"])
    p.add_argument("--max-chars", type=int, default=config["max_chars"], help="Split sentences longer than this")
    p.add_argument("--sentence-pause", type=float, default=config["sentence_pause"])
    p.add_argument("--paragraph-pause", type=float, default=config["paragraph_pause"])
    p.add_argument("--regen", default="", help="Sentence numbers to regenerate, e.g. 3,7")
    p.add_argument("--force", action="store_true", help="Regenerate every sentence")
    p.add_argument("--mp3", action=argparse.BooleanOptionalAction, default=config["mp3"],
                   help="Also export an MP3 (use --no-mp3 to skip)")
    p.add_argument("--set-default-voice", metavar="WAV", help="Save this reference voice to the config and exit")
    args = p.parse_args(argv)
    if not args.set_default_voice and args.text is None and not args.text_file and not args.batch_dir:
        p.error("provide --text, --text-file or --batch-dir")
    return args


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


def collect_jobs(args) -> list[tuple[str, str]]:
    """Return (script name, raw text) pairs for everything to narrate."""
    if args.text is not None:
        return [("narration", args.text)]
    if args.batch_dir:
        folder = Path(args.batch_dir)
        if not folder.is_dir():
            raise VoiceEngineError(f"Batch folder not found: {folder}")
        paths = sorted(folder.glob("*.txt"))
        if not paths:
            raise VoiceEngineError(f"No .txt scripts in {folder}")
    else:
        paths = [Path(f) for f in args.text_file]
    jobs = []
    for path in paths:
        if not path.is_file():
            raise VoiceEngineError(f"Text file not found: {path}")
        jobs.append((path.stem, path.read_text(encoding="utf-8")))
    return jobs


def run_job(stem, raw, args, get_model, pronunciations) -> bool:
    started = time.time()
    output = Path(args.output) if args.output else (
        Path(args.output_dir) / f"{stem}_{datetime.now():%Y%m%d-%H%M%S}.wav")
    work_dir = Path(args.work_dir) if args.work_dir else Path(args.output_dir) / "work" / stem
    log_path = output.parent / LOG_NAME
    record = {"script": stem, "voice": str(args.voice), "model": args.model, "seed": args.seed}
    try:
        text = normalize_text(raw, pronunciations)
        chunks = split_into_chunks(text, args.max_chars)
        if not chunks:
            raise VoiceEngineError("The script is empty.")
        regen = _parse_regen(args.regen, len(chunks))

        print(f"\n{len(chunks)} sentences:")
        for n, c in enumerate(chunks, start=1):
            print(f"{n:>4}. {c.text[:80]}{'...' if len(c.text) > 80 else ''}")
        print()

        paths = generate_chunks(get_model, chunks, args.voice, args.model, work_dir,
                                args.seed, regen, args.force)
        audio, sr = assemble(paths, chunks, args.sentence_pause, args.paragraph_pause)

        output.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output), np.clip(audio, -1.0, 1.0), sr, subtype="PCM_16")
        seconds = len(audio) / sr
        print(f"Saved {output} ({seconds:.1f}s of audio, {time.time() - started:.1f}s total)")
        mp3 = None
        if args.mp3:
            mp3 = export_mp3(output, output.with_suffix(".mp3"))
            print(f"Saved {mp3}")
        log_event(log_path, status="ok", sentences=len(chunks), audio_seconds=round(seconds, 1),
                  elapsed_seconds=round(time.time() - started, 1), output=str(output),
                  mp3=str(mp3) if mp3 else None, **record)
        return True
    except (VoiceEngineError, AudioError, TextError) as exc:
        print(f"Error in '{stem}': {exc}", file=sys.stderr)
        log_event(log_path, status="error", error=str(exc),
                  elapsed_seconds=round(time.time() - started, 1), **record)
        return False


def main(argv=None) -> int:
    try:
        args = parse_args(argv)
    except ConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        if args.set_default_voice:
            validate_reference(args.set_default_voice)
            config = load_config(args.config)
            config["voice"] = args.set_default_voice
            save_config(config, args.config)
            print(f"Default voice saved to {args.config}: {args.set_default_voice}")
            return 0

        report = validate_reference(args.voice)
        print(f"Reference: {report.seconds:.1f}s, {report.sample_rate} Hz, "
              f"{report.channels} ch, peak {report.peak:.2f}, RMS {report.rms_db:.0f} dBFS")
        for w in report.warnings:
            print(f"  warning: {w}")

        jobs = collect_jobs(args)
        if args.output and len(jobs) > 1:
            raise VoiceEngineError("--output works with a single script; use --output-dir for batches.")
        pronunciations = load_pronunciations(args.pronunciations)
    except (VoiceEngineError, AudioError, TextError, ConfigError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    models = {}

    def get_model():
        if "m" not in models:
            device = pick_device()
            print(f"Loading {args.model} model on {device}...")
            t0 = time.time()
            models["m"] = load_model(args.model, device)
            print(f"Model ready in {time.time() - t0:.1f}s")
        return models["m"]

    failures = 0
    for i, (stem, raw) in enumerate(jobs, start=1):
        if len(jobs) > 1:
            print(f"\n=== [{i}/{len(jobs)}] {stem} ===")
        if not run_job(stem, raw, args, get_model, pronunciations):
            failures += 1
    if len(jobs) > 1:
        print(f"\nBatch finished: {len(jobs) - failures} ok, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
