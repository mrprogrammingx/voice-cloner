# voice-narrator

Local voice-cloning narration for programming tutorials. Runs entirely on this
Mac: no recurring fees, and your recordings never leave the machine.

## How it works

The default engine (`vc`) works in two stages for every sentence:

1. Chatterbox's original model speaks the sentence in its built-in default
   voice (clean English pronunciation).
2. Chatterbox's voice conversion re-voices that audio to match your reference
   recording.

Why not clone directly? In testing, the faster Turbo model cloned this
reference clip with unpredictable accents. The two-stage route fixed the
pronunciation at the cost of some similarity and speed (about 7x slower than
real time on an M2 Pro with 16 GB). The `turbo` and `original` engines are
still available with `--model`.

## Licenses (checked 2026-10-28, re-check before relying on them)

- Chatterbox code: MIT (GitHub repository).
- Chatterbox weights: MIT (Hugging Face model card). The voice-conversion
  weights were not checked separately.
- Every output carries Resemble AI's imperceptible Perth watermark. It is not
  removed by this project.
- Only clone voices you have permission to use. Check YouTube's current policy
  on disclosing realistic synthetic audio.

## Setup from scratch

Needs Python 3.11, ffmpeg and git (Homebrew).

    cd ~/voice-narrator
    /opt/homebrew/bin/python3.11 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

`setuptools` must stay below version 81. Newer versions remove `pkg_resources`,
which the watermarking library imports; without it the model fails to load with
`'NoneType' object is not callable`.

The first run downloads model weights (about 3 GB) to `~/.cache/huggingface`.

## Everyday use

    source .venv/bin/activate
    python -m src.main --text-file scripts/tutorial.txt

With no other options this uses `config.json`: the saved voice, the `vc`
engine, seed 1, pauses and MP3 export. The output is
`output/<script name>_<date-time>.wav` (+ `.mp3`).

Explicit output name:

    python -m src.main --voice samples/ref_480.wav --text-file scripts/tutorial.txt --output output/tutorial.wav

Several scripts, or a whole folder (the model loads once; a failing script does
not stop the others):

    python -m src.main --text-file a.txt b.txt
    python -m src.main --batch-dir scripts/

Save a different default voice (it must be a clean .wav or .flac of about
10-30 seconds):

    python -m src.main --set-default-voice samples/ref_480.wav

Cut a reference clip from a long recording of any format:

    ffmpeg -ss 480 -t 20 -i samples/recording.m4a -ac 1 -c:a pcm_s16le samples/ref_480.wav

## Settings (`config.json`)

voice, model (`vc`, `turbo` or `original`), seed, max_chars, sentence_pause,
paragraph_pause, pronunciations, output_dir, mp3. Command-line options override
the file. Unknown or wrongly typed settings are rejected with a clear message.

## Fixing individual sentences

Every run prints a numbered sentence list. Sentences are cached in
`output/work/<script name>/`, so re-running only generates what changed.

    # redo sentences 3 and 7 with a new random take
    python -m src.main --text-file scripts/tutorial.txt --regen 3,7

    # audition 4 takes of sentence 3, listen, keep the best, rebuild
    python -m src.takes --voice samples/ref_480.wav --text-file scripts/tutorial.txt \
        --output output/tutorial.wav --sentence 3 --takes 4
    afplay output/work/tutorial/takes/chunk_0003_take2.wav
    python -m src.takes --voice samples/ref_480.wav --text-file scripts/tutorial.txt \
        --output output/tutorial.wav --sentence 3 --pick 2
    python -m src.main --text-file scripts/tutorial.txt

Use the same script name, voice, model and seed for `takes` and `main`, or the
cache will not match. Changing a sentence's text, the voice, the model or the
seed regenerates it. If you change a pronunciation, do it before auditioning.

## Pronunciation and script tips

- Inline `code` (backticks) is spoken as words: `get_user_name` becomes
  "get user name"; fenced code blocks are skipped.
- `pronunciations.json` respells words. Respelling is unreliable with this
  model (it hurt "Python" and "data" in testing); it works best for odd tool
  names. Fixing a bad word usually means auditioning more takes.
- Blank lines start a new paragraph (longer pause).

## Web interface

    python -m src.app

Opens http://127.0.0.1:7860 (this Mac only, no public link, analytics off).
**Stop it with Ctrl+C when you are done**: it keeps the model in memory, which
slows everything else on a 16 GB Mac.

## Performance and memory

- Expect roughly 7x real time with the `vc` engine: a 10-minute narration can
  take over an hour. Cached sentences make re-runs cheap.
- Quit VS Code, OBS and other heavy apps for long jobs. Heavy swap (check with
  `sysctl vm.swapusage`) hurts speed.
- `python -m scripts.memory_check --model vc --rounds 4` repeats generations
  and prints memory use. In testing it stayed flat (no leak).

## Tests

    python -m pytest tests -q

The tests use a fake model: they take seconds and need no GPU or downloads.
They check our code, not the voice quality.

Tell a bug from a model limitation:

| Symptom | Likely cause |
|---|---|
| Tests fail, or the same input always crashes | Bug in this code |
| Wrong cache behaviour, missing files, wrong pauses | Bug |
| A word is wrong on one take but fine on another | Model randomness: audition takes |
| Accent or similarity is off on every take | Model/reference limitation |

## Safe updates

Never upgrade packages directly in `.venv`. Use:

    ./scripts/safe_update.sh          # test new versions in .venv-test
    ./scripts/safe_update.sh smoke    # also do one real generation

Adopt only if the tests pass, and listen to a short script afterwards.

Model weights: while online, Hugging Face may fetch updated weights the next
time a model loads. To freeze the weights you have now, run with
`HF_HUB_OFFLINE=1`:

    HF_HUB_OFFLINE=1 python -m src.main --text-file scripts/tutorial.txt

This fails with an error if a needed file is not in the cache. Chatterbox's API
does not expose a way to pin a model revision that was verified here.

## Logs and privacy

`output/generation_log.jsonl` records script names, settings, timings and
output paths. It never stores script text. Git ignores `samples/`, `models/`,
`output/` and all audio files; never commit voice recordings or weights.

## Layout

    src/main.py          command line (batch, config, log)
    src/pipeline.py      per-sentence generation, cache, assembly
    src/voice_engine.py  model loading and synthesis
    src/vc_engine.py     two-stage default voice + conversion engine
    src/text_utils.py    normalization, pronunciations, chunking
    src/audio_utils.py   reference checks, ffmpeg conversion, MP3
    src/takes.py         audition and pick takes
    src/app.py           local web interface
    scripts/             experiments and tools (benchmarks, safe_update.sh)
    tests/               pytest suite
