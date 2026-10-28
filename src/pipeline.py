"""Chunked narration: one clip per sentence, cached on disk, assembled with pauses."""
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio_utils import AudioError
from .text_utils import Chunk
from .voice_engine import synthesize

MANIFEST = "manifest.json"


def _chunk_path(work_dir: Path, number: int) -> Path:
    return work_dir / f"chunk_{number:04d}.wav"


def _load_manifest(work_dir: Path) -> dict:
    path = work_dir / MANIFEST
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def generate_chunks(get_model, chunks: list[Chunk], voice, kind: str, work_dir,
                    base_seed: int | None = 1, regen=frozenset(), force: bool = False,
                    log=print) -> list[Path]:
    """Generate (or reuse) one WAV per chunk. `regen` holds 1-based chunk numbers to redo."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    settings = {"voice": str(voice), "model": kind, "seed": base_seed}

    manifest = _load_manifest(work_dir)
    same_settings = all(manifest.get(k) == v for k, v in settings.items())
    prior = manifest.get("chunks", []) if same_settings else []

    entries, paths = [], []
    total = len(chunks)
    for n, chunk in enumerate(chunks, start=1):
        path = _chunk_path(work_dir, n)
        old = prior[n - 1] if n <= len(prior) else None
        same_text = old is not None and old["text"] == chunk.text and path.is_file()
        attempt = old["attempt"] if same_text else 0

        if same_text and n not in regen and not force:
            log(f"[{n}/{total}] reused")
        else:
            if same_text and n in regen:
                attempt += 1
            seed = None if base_seed is None else base_seed + n + 1000 * attempt
            start = time.time()
            model = get_model()
            wav = synthesize(model, chunk.text, voice, kind, seed)
            audio = wav.squeeze().detach().cpu().numpy().astype("float32")
            sf.write(str(path), audio, model.sr, subtype="FLOAT")
            log(f"[{n}/{total}] generated {len(audio) / model.sr:.1f}s in {time.time() - start:.1f}s (attempt {attempt})")

        entries.append({"text": chunk.text, "paragraph_end": chunk.paragraph_end, "attempt": attempt})
        paths.append(path)
        # Save progress after every chunk so an interrupted run can resume.
        manifest_out = {**settings, "chunks": entries + prior[len(entries):]}
        (work_dir / MANIFEST).write_text(json.dumps(manifest_out, indent=2), encoding="utf-8")
    return paths


def _trim_silence(audio: np.ndarray, sr: int, threshold_db: float = -45.0, keep_ms: int = 40) -> np.ndarray:
    loud = np.flatnonzero(np.abs(audio) > 10 ** (threshold_db / 20))
    if loud.size == 0:
        return audio
    keep = int(sr * keep_ms / 1000)
    return audio[max(0, loud[0] - keep): min(len(audio), loud[-1] + keep)]


def _match_level(audio: np.ndarray, target_db: float) -> np.ndarray:
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < 1e-6:
        return audio
    gain = 10 ** (target_db / 20) / rms
    peak = float(np.max(np.abs(audio)))
    if peak * gain > 0.97:
        gain = 0.97 / peak
    return audio * gain


def _fade(audio: np.ndarray, sr: int, ms: int = 10) -> np.ndarray:
    n = min(int(sr * ms / 1000), len(audio) // 2)
    if n == 0:
        return audio
    ramp = np.linspace(0.0, 1.0, n, dtype="float32")
    audio = audio.copy()
    audio[:n] *= ramp
    audio[-n:] *= ramp[::-1]
    return audio


def assemble(paths: list[Path], chunks: list[Chunk], sentence_pause: float = 0.35,
             paragraph_pause: float = 0.8, target_db: float = -20.0) -> tuple[np.ndarray, int]:
    """Trim, level-match, fade and join chunk clips with pauses. Returns (audio, sample_rate)."""
    if not paths:
        raise AudioError("No audio chunks to assemble.")
    pieces, sr = [], None
    for i, (path, chunk) in enumerate(zip(paths, chunks)):
        audio, file_sr = sf.read(str(path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr is None:
            sr = file_sr
        elif file_sr != sr:
            raise AudioError(f"Sample rate mismatch in {path} ({file_sr} vs {sr}).")
        pieces.append(_fade(_match_level(_trim_silence(audio, sr), target_db), sr))
        if i < len(paths) - 1:
            pause = paragraph_pause if chunk.paragraph_end else sentence_pause
            pieces.append(np.zeros(int(pause * sr), dtype="float32"))
    return np.concatenate(pieces), sr
