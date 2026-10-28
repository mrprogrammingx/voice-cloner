"""Command-line entry point: python -m src.main --voice ... --text ... --output ..."""
import argparse
import sys
import time
from pathlib import Path

from .audio_utils import AudioError, validate_reference
from .voice_engine import (MODEL_KINDS, VoiceEngineError, load_model, pick_device,
                           save_wav, synthesize)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Generate narration in a cloned voice.")
    p.add_argument("--voice", required=True, help="Reference voice (.wav or .flac)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="Script text")
    src.add_argument("--text-file", help="Path to a UTF-8 text file with the script")
    p.add_argument("--output", default="output/narration.wav", help="Output WAV path")
    p.add_argument("--model", choices=MODEL_KINDS, default="turbo")
    p.add_argument("--seed", type=int, default=None, help="Random seed (results vary without one)")
    p.add_argument("--exaggeration", type=float, default=0.5, help="original model only")
    p.add_argument("--cfg-weight", type=float, default=0.5, help="original model only")
    return p.parse_args(argv)


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
            text = text_path.read_text(encoding="utf-8")
        else:
            text = args.text

        device = pick_device()
        print(f"Loading {args.model} model on {device}...")
        t0 = time.time()
        model = load_model(args.model, device)
        print(f"Model ready in {time.time() - t0:.1f}s")

        print("Generating...")
        t1 = time.time()
        wav = synthesize(model, text, args.voice, args.model, args.seed,
                         args.exaggeration, args.cfg_weight)
        elapsed = time.time() - t1

        out = save_wav(wav, model.sr, args.output)
        seconds = wav.shape[-1] / model.sr
        print(f"Saved {out} ({seconds:.1f}s of audio, generated in {elapsed:.1f}s)")
        return 0
    except (VoiceEngineError, AudioError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
