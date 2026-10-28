"""Script preparation: normalization, pronunciation overrides, chunking."""
import json
import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PRONUNCIATIONS_PATH = Path("pronunciations.json")

_REPLACEMENTS = {
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": ", ", "\u2026": "...", "&": " and ",
}
_ABBREVIATIONS = ("e.g.", "i.e.", "vs.", "Dr.", "Mr.", "Ms.")
_DOT = "\x00"


class TextError(Exception):
    """Raised when the pronunciation file or script text is unusable."""


@dataclass
class Chunk:
    text: str
    paragraph_end: bool  # True if a paragraph break follows this chunk


def load_pronunciations(path: str | Path = DEFAULT_PRONUNCIATIONS_PATH) -> dict[str, str]:
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TextError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise TextError(f"{path} must contain a JSON object like {{\"word\": \"spoken form\"}}")
    return {str(k): str(v) for k, v in data.items()}


def _split_identifier(token: str) -> str:
    token = token.replace("_", " ")
    token = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", token)
    token = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", token)
    return token.strip()


def normalize_text(text: str, pronunciations: dict[str, str] | None = None) -> str:
    """Clean a script for speech. Paragraph breaks (blank lines) are preserved."""
    for old, new in _REPLACEMENTS.items():
        text = text.replace(old, new)
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)  # drop fenced code blocks
    text = re.sub(r"`([^`]+)`", lambda m: _split_identifier(m.group(1)), text)

    for word in sorted((pronunciations or {}), key=len, reverse=True):
        text = re.sub(rf"(?<!\w){re.escape(word)}(?!\w)", pronunciations[word], text)

    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    return "\n\n".join(p for p in paragraphs if p)


def _split_sentences(paragraph: str) -> list[str]:
    for abbr in _ABBREVIATIONS:
        paragraph = paragraph.replace(abbr, abbr.replace(".", _DOT))
    parts = re.split(r"(?<=[.!?])\s+", paragraph)
    return [p.replace(_DOT, ".").strip() for p in parts if p.strip()]


def _split_long(sentence: str, max_chars: int) -> list[str]:
    pieces, current = [], ""
    for part in re.split(r"(?<=[,;:])\s+", sentence):
        if current and len(current) + 1 + len(part) > max_chars:
            pieces.append(current)
            current = part
        else:
            current = f"{current} {part}".strip()
    pieces.append(current)

    result = []
    for piece in pieces:
        while len(piece) > max_chars:
            cut = piece.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            result.append(piece[:cut].strip())
            piece = piece[cut:].strip()
        if piece:
            result.append(piece)
    return result


def split_into_chunks(text: str, max_chars: int = 250) -> list[Chunk]:
    """Split normalized text into sentence-sized chunks, remembering paragraph ends."""
    chunks: list[Chunk] = []
    for paragraph in re.split(r"\n\s*\n", text.strip()):
        pieces = []
        for sentence in _split_sentences(re.sub(r"\s+", " ", paragraph)):
            pieces.extend(_split_long(sentence, max_chars) if len(sentence) > max_chars else [sentence])
        for i, piece in enumerate(pieces):
            chunks.append(Chunk(piece, paragraph_end=(i == len(pieces) - 1)))
    return chunks
