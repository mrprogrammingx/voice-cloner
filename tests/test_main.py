import shutil

import pytest

from src.main import main


def args(voice, *extra):
    return ["--voice", str(voice), "--pronunciations", "no-such-file.json", *extra]


def test_missing_voice(capsys, tmp_path):
    assert main(args(tmp_path / "nope.wav", "--text", "Hello.")) == 1
    assert "not found" in capsys.readouterr().err


def test_unsupported_audio_format(capsys, tmp_path):
    p = tmp_path / "voice.m4a"
    p.write_bytes(b"x")
    assert main(args(p, "--text", "Hello.")) == 1
    assert "Unsupported format" in capsys.readouterr().err


def test_empty_script(capsys, voice_wav):
    assert main(args(voice_wav, "--text", "   ")) == 1
    assert "empty" in capsys.readouterr().err


def test_missing_text_file(capsys, voice_wav, tmp_path):
    assert main(args(voice_wav, "--text-file", str(tmp_path / "nope.txt"))) == 1
    assert "not found" in capsys.readouterr().err


def test_regen_number_out_of_range(capsys, voice_wav):
    assert main(args(voice_wav, "--text", "Hello there.", "--regen", "5")) == 1
    assert "out of range" in capsys.readouterr().err


def test_full_run_creates_output(monkeypatch, tmp_path, fake_model, voice_wav):
    monkeypatch.setattr("src.main.load_model", lambda kind, device: fake_model)
    out = tmp_path / "out" / "narration.wav"
    code = main(args(voice_wav, "--text", "One. Two.", "--output", str(out),
                     "--work-dir", str(tmp_path / "work")))
    assert code == 0 and out.is_file() and out.stat().st_size > 0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_full_run_with_mp3(monkeypatch, tmp_path, fake_model, voice_wav):
    monkeypatch.setattr("src.main.load_model", lambda kind, device: fake_model)
    out = tmp_path / "narration.wav"
    code = main(args(voice_wav, "--text", "One. Two.", "--output", str(out),
                     "--work-dir", str(tmp_path / "work"), "--mp3"))
    assert code == 0 and out.with_suffix(".mp3").is_file()


def test_repeated_runs_do_not_regenerate(monkeypatch, tmp_path, fake_model, voice_wav):
    monkeypatch.setattr("src.main.load_model", lambda kind, device: fake_model)
    common = args(voice_wav, "--text", "One. Two.", "--output", str(tmp_path / "n.wav"),
                  "--work-dir", str(tmp_path / "work"))
    assert main(common) == 0 and main(common) == 0
    assert len(fake_model.calls) == 2
