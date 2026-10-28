"""Local web interface: python -m src.app  (serves http://127.0.0.1:7860 only)."""
import gc
import re
import time
from datetime import datetime
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf
import torch

from .audio_utils import AudioError, convert_to_wav, export_mp3, validate_reference
from .main import _parse_regen
from .pipeline import assemble, generate_chunks
from .text_utils import TextError, load_pronunciations, normalize_text, split_into_chunks
from .voice_engine import MODEL_KINDS, VoiceEngineError, load_model, pick_device

SAMPLES = Path("samples")
OUTPUT = Path("output")

# Only one model is kept in memory at a time (16 GB RAM is tight).
_model_state = {"kind": None, "model": None}


def _get_model(kind: str):
    if _model_state["kind"] != kind:
        _model_state["model"] = None
        _model_state["kind"] = None
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        _model_state["model"] = load_model(kind, pick_device())
        _model_state["kind"] = kind
    return _model_state["model"]


def list_voices() -> list[str]:
    SAMPLES.mkdir(exist_ok=True)
    return sorted(p.name for p in SAMPLES.iterdir() if p.suffix.lower() in {".wav", ".flac"})


def resolve_voice(selected, upload, start, length) -> Path:
    if upload:
        dst = SAMPLES / f"upload_{datetime.now():%Y%m%d_%H%M%S}.wav"
        return convert_to_wav(upload, dst, start=float(start), duration=float(length))
    if not selected:
        raise AudioError("Choose a saved voice or upload a recording.")
    return SAMPLES / selected


def generate(script, selected, upload, start, length, kind, seed, max_chars,
             sentence_pause, paragraph_pause, name, regen_text, progress=gr.Progress()):
    t_start = time.time()
    try:
        progress(0, desc="Checking reference voice...")
        voice = resolve_voice(selected, upload, start, length)
        report = validate_reference(voice)

        text = normalize_text(script or "", load_pronunciations())
        chunks = split_into_chunks(text, int(max_chars))
        if not chunks:
            raise VoiceEngineError("The script is empty.")
        regen = _parse_regen(regen_text or "", len(chunks))
        listing = "\n".join(f"{n}. {c.text}" for n, c in enumerate(chunks, start=1))

        stem = re.sub(r"[^\w\-]+", "_", (name or "").strip()).strip("_") or "narration"
        output = OUTPUT / f"{stem}.wav"
        work_dir = OUTPUT / "work" / stem

        def get_model():
            if _model_state["kind"] != kind:
                progress(0.05, desc=f"Loading {kind} model (about 20s the first time)...")
            return _get_model(kind)

        def log(msg):
            m = re.match(r"\[(\d+)/(\d+)\]", msg)
            if m:
                progress(0.1 + 0.8 * int(m[1]) / int(m[2]), desc=f"Sentence {msg}")

        paths = generate_chunks(get_model, chunks, str(voice), kind, work_dir, int(seed), regen, False, log)

        progress(0.92, desc="Assembling audio...")
        audio, sr = assemble(paths, chunks, float(sentence_pause), float(paragraph_pause))
        output.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output), np.clip(audio, -1.0, 1.0), sr, subtype="PCM_16")

        progress(0.97, desc="Exporting MP3...")
        mp3 = export_mp3(output, output.with_suffix(".mp3"))

        notes = "".join(f"\nWarning: {w}" for w in report.warnings)
        status = (f"Done: {len(audio) / sr:.1f}s of audio from {len(chunks)} sentences "
                  f"in {time.time() - t_start:.1f}s.\nSaved {output} and {mp3}{notes}")
        return str(output), [str(output), str(mp3)], listing, status
    except (VoiceEngineError, AudioError, TextError) as exc:
        raise gr.Error(str(exc))
    except Exception as exc:
        raise gr.Error(f"Unexpected error: {type(exc).__name__}: {exc}")


def build_ui():
    voices = list_voices()
    default = "ref_480.wav" if "ref_480.wav" in voices else (voices[0] if voices else None)

    with gr.Blocks(title="Voice Narrator", analytics_enabled=False) as demo:
        gr.Markdown("# Voice Narrator\nRuns entirely on this Mac. Nothing is sent to any service.")
        with gr.Row():
            with gr.Column(scale=2):
                script = gr.Textbox(label="Script", lines=16,
                                    placeholder="Paste your English narration script here. Blank lines start a new paragraph.")
                regen = gr.Textbox(label="Regenerate sentences (optional)",
                                   info="Sentence numbers from the list below, e.g. 3,7. Leave empty to reuse existing sentences.")
                generate_btn = gr.Button("Generate", variant="primary")
                status = gr.Textbox(label="Status", lines=3, interactive=False)
            with gr.Column(scale=1):
                voice = gr.Dropdown(choices=voices, value=default, label="Saved voice (from samples/)")
                refresh = gr.Button("Refresh voice list")
                upload = gr.Audio(sources=["upload"], type="filepath",
                                  label="...or upload a recording (overrides the saved voice)")
                start = gr.Slider(minimum=0, maximum=900, value=0, step=1, label="Upload: start at (seconds)")
                length = gr.Slider(minimum=5, maximum=40, value=20, step=1, label="Upload: clip length (seconds)")
                with gr.Accordion("Settings", open=False):
                    model = gr.Dropdown(choices=list(MODEL_KINDS), value="vc", label="Model")
                    seed = gr.Slider(minimum=0, maximum=9999, value=1, step=1, label="Seed")
                    max_chars = gr.Slider(minimum=80, maximum=400, value=250, step=10, label="Max characters per chunk")
                    s_pause = gr.Slider(minimum=0, maximum=2, value=0.35, step=0.05, label="Pause between sentences (s)")
                    p_pause = gr.Slider(minimum=0, maximum=3, value=0.8, step=0.05, label="Pause between paragraphs (s)")
                    name = gr.Textbox(value="narration", label="Output name")
        player = gr.Audio(label="Preview", type="filepath", interactive=False)
        files = gr.File(label="Download (WAV and MP3)", file_count="multiple", interactive=False)
        sentences = gr.Textbox(label="Sentences (numbers for regeneration)", lines=8, interactive=False)

        refresh.click(lambda current: gr.Dropdown(choices=list_voices(), value=current),
                      inputs=voice, outputs=voice)
        generate_btn.click(
            generate,
            inputs=[script, voice, upload, start, length, model, seed, max_chars, s_pause, p_pause, name, regen],
            outputs=[player, files, sentences, status],
        )
    return demo


def main():
    demo = build_ui()
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name="127.0.0.1",   # this Mac only
        server_port=7860,
        share=False,               # never create a public link
        show_error=True,
        inbrowser=True,
        allowed_paths=[str(OUTPUT.resolve())],
    )


if __name__ == "__main__":
    main()
