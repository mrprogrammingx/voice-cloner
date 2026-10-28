import shutil

import numpy as np
import pytest
import soundfile as sf

from src.audio_utils import AudioError, convert_to_wav, export_mp3, validate_reference

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def write_tone(path, seconds=10, amplitude=0.2, sr=24000):
    t = np.arange(int(sr * seconds)) / sr
    sf.write(str(path), (amplitude * np.sin(2 * np.pi * 220 * t)).astype("float32"), sr)


def test_missing_reference(tmp_path):
    with pytest.raises(AudioError, match="not found"):
        validate_reference(tmp_path / "nope.wav")


def test_unsupported_format(tmp_path):
    p = tmp_path / "voice.m4a"
    p.write_bytes(b"x")
    with pytest.raises(AudioError, match="Unsupported format"):
        validate_reference(p)


def test_corrupt_wav(tmp_path):
    p = tmp_path / "bad.wav"
    p.write_bytes(b"this is not audio")
    with pytest.raises(AudioError, match="Could not read"):
        validate_reference(p)


def test_too_short_reference(tmp_path):
    p = tmp_path / "short.wav"
    write_tone(p, seconds=1)
    with pytest.raises(AudioError, match="only"):
        validate_reference(p)


def test_valid_reference_has_no_warnings(voice_wav):
    report = validate_reference(voice_wav)
    assert report.warnings == []
    assert report.seconds == pytest.approx(10, abs=0.01)


def test_quiet_reference_warns(tmp_path):
    p = tmp_path / "quiet.wav"
    write_tone(p, amplitude=0.001)
    assert any("quiet" in w for w in validate_reference(p).warnings)


def test_clipping_reference_warns(tmp_path):
    p = tmp_path / "loud.wav"
    write_tone(p, amplitude=1.0)
    assert any("clipping" in w for w in validate_reference(p).warnings)


@needs_ffmpeg
def test_convert_to_wav_cuts_a_clip(tmp_path, voice_wav):
    out = convert_to_wav(voice_wav, tmp_path / "clip.wav", start=1, duration=2)
    data, sr = sf.read(str(out))
    assert len(data) / sr == pytest.approx(2, abs=0.05)


def test_convert_missing_input(tmp_path):
    with pytest.raises(AudioError, match="not found"):
        convert_to_wav(tmp_path / "nope.m4a", tmp_path / "x.wav")


@needs_ffmpeg
def test_export_mp3_creates_file(tmp_path, voice_wav):
    mp3 = export_mp3(voice_wav, tmp_path / "out.mp3")
    assert mp3.is_file() and mp3.stat().st_size > 0
