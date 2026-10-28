import json

import numpy as np
import pytest
import soundfile as sf

from src.audio_utils import AudioError
from src.pipeline import assemble, generate_chunks
from src.text_utils import split_into_chunks


def run(model, chunks, voice, work_dir, seed=1, **kw):
    loads = []

    def get_model():
        loads.append(1)
        return model

    paths = generate_chunks(get_model, chunks, str(voice), "turbo", work_dir, seed, log=lambda m: None, **kw)
    return paths, loads


def test_generates_files_and_manifest(tmp_path, fake_model, voice_wav):
    chunks = split_into_chunks("One. Two. Three.")
    paths, _ = run(fake_model, chunks, voice_wav, tmp_path / "work")
    assert len(paths) == 3 and all(p.is_file() for p in paths)
    assert (tmp_path / "work" / "manifest.json").is_file()
    assert len(fake_model.calls) == 3


def test_repeated_run_reuses_everything_and_loads_no_model(tmp_path, fake_model, voice_wav):
    chunks = split_into_chunks("One. Two. Three.")
    run(fake_model, chunks, voice_wav, tmp_path / "work")
    _, loads = run(fake_model, chunks, voice_wav, tmp_path / "work")
    assert loads == [] and len(fake_model.calls) == 3


def test_regen_redoes_only_selected_sentence(tmp_path, fake_model, voice_wav):
    chunks = split_into_chunks("One. Two. Three.")
    run(fake_model, chunks, voice_wav, tmp_path / "work")
    run(fake_model, chunks, voice_wav, tmp_path / "work", regen={2})
    assert len(fake_model.calls) == 4
    manifest = json.loads((tmp_path / "work" / "manifest.json").read_text())
    assert [c["attempt"] for c in manifest["chunks"]] == [0, 1, 0]


def test_changed_text_regenerates_only_that_sentence(tmp_path, fake_model, voice_wav):
    run(fake_model, split_into_chunks("One. Two. Three."), voice_wav, tmp_path / "work")
    run(fake_model, split_into_chunks("One. Changed. Three."), voice_wav, tmp_path / "work")
    assert fake_model.calls[-1] == "Changed." and len(fake_model.calls) == 4


def test_changed_seed_regenerates_everything(tmp_path, fake_model, voice_wav):
    chunks = split_into_chunks("One. Two.")
    run(fake_model, chunks, voice_wav, tmp_path / "work", seed=1)
    run(fake_model, chunks, voice_wav, tmp_path / "work", seed=2)
    assert len(fake_model.calls) == 4


def test_force_regenerates_everything(tmp_path, fake_model, voice_wav):
    chunks = split_into_chunks("One. Two.")
    run(fake_model, chunks, voice_wav, tmp_path / "work")
    run(fake_model, chunks, voice_wav, tmp_path / "work", force=True)
    assert len(fake_model.calls) == 4


def test_very_long_script(tmp_path, fake_model, voice_wav):
    text = " ".join(f"Sentence number {i}." for i in range(300))
    chunks = split_into_chunks(text)
    paths, _ = run(fake_model, chunks, voice_wav, tmp_path / "work")
    assert len(chunks) == 300 and all(p.is_file() for p in paths)


def test_assemble_length_includes_pauses(tmp_path, fake_model, voice_wav):
    chunks = split_into_chunks("A one. B two.\n\nC three.")
    paths, _ = run(fake_model, chunks, voice_wav, tmp_path / "work")
    audio, sr = assemble(paths, chunks, sentence_pause=0.35, paragraph_pause=0.8)
    assert len(audio) / sr == pytest.approx(3 * 0.5 + 0.35 + 0.8, abs=0.02)


def test_assemble_has_no_clicks_or_nans(tmp_path, fake_model, voice_wav):
    chunks = split_into_chunks("A one. B two. C three.")
    paths, _ = run(fake_model, chunks, voice_wav, tmp_path / "work")
    audio, _ = assemble(paths, chunks)
    assert not np.isnan(audio).any()
    assert abs(audio[0]) < 1e-3 and abs(audio[-1]) < 1e-3
    assert np.max(np.abs(np.diff(audio))) < 0.05  # no sudden jumps


def test_assemble_rejects_empty_input():
    with pytest.raises(AudioError, match="No audio"):
        assemble([], [])


def test_assemble_rejects_mixed_sample_rates(tmp_path):
    chunks = split_into_chunks("One. Two.")
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    sf.write(str(a), np.full(8000, 0.1, dtype="float32"), 16000)
    sf.write(str(b), np.full(8000, 0.1, dtype="float32"), 24000)
    with pytest.raises(AudioError, match="Sample rate mismatch"):
        assemble([a, b], chunks)
