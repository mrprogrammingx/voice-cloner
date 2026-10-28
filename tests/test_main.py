import json
import shutil

import pytest

from src.main import main


@pytest.fixture(autouse=True)
def run_in_tmp(tmp_path, monkeypatch):
    """Relative paths like output/ resolve inside the test's temp folder."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def patched_model(monkeypatch, fake_model):
    loads = []

    def fake_load(kind, device):
        loads.append(kind)
        return fake_model

    monkeypatch.setattr("src.main.load_model", fake_load)
    return fake_model, loads


def args(voice, *extra):
    return ["--config", "no-such-config.json", "--voice", str(voice),
            "--pronunciations", "no-such-file.json", "--no-mp3", *extra]


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


def test_bad_config_file_is_reported(capsys, tmp_path, voice_wav):
    bad = tmp_path / "bad.json"
    bad.write_text("{nope")
    assert main(["--config", str(bad), "--voice", str(voice_wav), "--text", "Hi."]) == 1
    assert "Invalid JSON" in capsys.readouterr().err


def test_full_run_creates_output(patched_model, tmp_path, voice_wav):
    out = tmp_path / "out" / "narration.wav"
    code = main(args(voice_wav, "--text", "One. Two.", "--output", str(out),
                     "--work-dir", str(tmp_path / "work")))
    assert code == 0 and out.is_file() and out.stat().st_size > 0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_full_run_with_mp3(patched_model, tmp_path, voice_wav):
    out = tmp_path / "narration.wav"
    code = main(args(voice_wav, "--text", "One. Two.", "--output", str(out),
                     "--work-dir", str(tmp_path / "work"), "--mp3"))
    assert code == 0 and out.with_suffix(".mp3").is_file()


def test_repeated_runs_do_not_regenerate(patched_model, tmp_path, voice_wav):
    model, _ = patched_model
    common = args(voice_wav, "--text", "One. Two.", "--output", str(tmp_path / "n.wav"),
                  "--work-dir", str(tmp_path / "work"))
    assert main(common) == 0 and main(common) == 0
    assert len(model.calls) == 2


def test_automatic_output_name(patched_model, tmp_path, voice_wav):
    script = tmp_path / "lesson1.txt"
    script.write_text("One. Two.")
    assert main(args(voice_wav, "--text-file", str(script), "--output-dir", str(tmp_path / "o"))) == 0
    assert len(list((tmp_path / "o").glob("lesson1_*.wav"))) == 1


def test_batch_loads_model_once_and_continues_after_failure(patched_model, tmp_path, voice_wav):
    model, loads = patched_model
    folder = tmp_path / "scripts"
    folder.mkdir()
    (folder / "a_good.txt").write_text("One. Two.")
    (folder / "b_empty.txt").write_text("   ")
    (folder / "c_good.txt").write_text("Three.")
    code = main(args(voice_wav, "--batch-dir", str(folder), "--output-dir", str(tmp_path / "o")))
    assert code == 1  # one script failed
    assert loads == ["vc"] or len(loads) == 1
    assert len(list((tmp_path / "o").glob("a_good_*.wav"))) == 1
    assert len(list((tmp_path / "o").glob("c_good_*.wav"))) == 1
    assert not list((tmp_path / "o").glob("b_empty_*.wav"))


def test_output_flag_rejected_for_batches(capsys, patched_model, tmp_path, voice_wav):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text("One.")
    b.write_text("Two.")
    code = main(args(voice_wav, "--text-file", str(a), str(b), "--output", str(tmp_path / "x.wav")))
    assert code == 1 and "single script" in capsys.readouterr().err


def test_generation_log_is_written_without_script_text(patched_model, tmp_path, voice_wav):
    secret = "Confidential sentence about nothing."
    out = tmp_path / "o" / "n.wav"
    assert main(args(voice_wav, "--text", secret, "--output", str(out))) == 0
    log = (out.parent / "generation_log.jsonl").read_text()
    record = json.loads(log.strip().splitlines()[-1])
    assert record["status"] == "ok" and record["sentences"] == 1
    assert "Confidential" not in log


def test_set_default_voice_is_used_next_time(patched_model, tmp_path, voice_wav):
    cfg = tmp_path / "cfg.json"
    assert main(["--config", str(cfg), "--set-default-voice", str(voice_wav)]) == 0
    assert json.loads(cfg.read_text())["voice"] == str(voice_wav)
    out = tmp_path / "n.wav"
    code = main(["--config", str(cfg), "--pronunciations", "none.json", "--no-mp3",
                 "--text", "One.", "--output", str(out)])
    assert code == 0 and out.is_file()


def test_set_default_voice_rejects_bad_file(capsys, tmp_path):
    cfg = tmp_path / "cfg.json"
    assert main(["--config", str(cfg), "--set-default-voice", str(tmp_path / "nope.wav")]) == 1
    assert not cfg.exists()
