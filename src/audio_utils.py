"""Audio helpers: reference validation and conversion."""
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

SUPPORTED_EXTENSIONS = {".wav", ".flac"}
MIN_SECONDS = 3.0      # below this the clip is rejected
GOOD_MIN_SECONDS = 8.0
GOOD_MAX_SECONDS = 30.0


class AudioError(Exception):
    """Raised when audio input is missing, unsupported, or unusable."""


@dataclass
class ReferenceReport:
    path: Path
    seconds: float
    sample_rate: int
    channels: int
    peak: float
    rms_db: float
    warnings: list[str] = field(default_factory=list)


def validate_reference(path: str | Path) -> ReferenceReport:
    """Check a reference clip. Raises AudioError for unusable files; warns otherwise."""
    path = Path(path)
    if not path.is_file():
        raise AudioError(f"Reference voice not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise AudioError(
            f"Unsupported format '{path.suffix}'. Convert it first, e.g.:\n"
            f"  ffmpeg -i {path} -ac 1 -c:a pcm_s16le samples/my_voice.wav"
        )
    try:
        data, sr = sf.read(str(path), always_2d=True)
    except Exception as exc:
        raise AudioError(f"Could not read '{path}': {exc}") from exc

    seconds = data.shape[0] / sr
    if seconds < MIN_SECONDS:
        raise AudioError(f"Reference is only {seconds:.1f}s; use at least {MIN_SECONDS:.0f}s of clean speech.")

    mono = data.mean(axis=1)
    peak = float(np.max(np.abs(mono)))
    rms = float(np.sqrt(np.mean(mono ** 2)))
    rms_db = 20 * np.log10(max(rms, 1e-9))

    warnings = []
    if seconds < GOOD_MIN_SECONDS:
        warnings.append(f"Short reference ({seconds:.1f}s); similarity may suffer.")
    if seconds > GOOD_MAX_SECONDS:
        warnings.append(f"Long reference ({seconds:.1f}s); a clean 10-30s clip is usually enough.")
    if peak >= 0.99:
        warnings.append("Reference is clipping (peak at full scale); re-record a bit quieter.")
    if rms_db < -35:
        warnings.append(f"Reference is very quiet (RMS {rms_db:.0f} dBFS); re-record closer to the mic.")
    if data.shape[1] > 1:
        warnings.append("Reference has multiple channels; they will be mixed to mono.")

    return ReferenceReport(path, seconds, sr, data.shape[1], peak, rms_db, warnings)


def convert_to_wav(src: str | Path, dst: str | Path, start: float = 0.0, duration: float | None = None) -> Path:
    """Convert any audio file ffmpeg can read to a mono 16-bit WAV."""
    src, dst = Path(src), Path(dst)
    if not src.is_file():
        raise AudioError(f"Input not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(start)]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd += ["-i", str(src), "-ac", "1", "-c:a", "pcm_s16le", str(dst)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise AudioError("ffmpeg is not installed or not on PATH.") from exc
    except subprocess.CalledProcessError as exc:
        raise AudioError(f"ffmpeg failed: {exc.stderr.strip()}") from exc
    return dst
