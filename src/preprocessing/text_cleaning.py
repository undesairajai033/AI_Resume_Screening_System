"""Text cleaning shared by every scoring method."""

import re
import unicodedata

# Keep characters that matter in tech skills: c++, c#, .net, node.js, ci/cd
_ALLOWED = re.compile(r"[^a-z0-9+#./\s-]")
_SPACES = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase, strip accents and stray symbols, collapse whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"https?://\S+|\S+@\S+", " ", text)   # URLs and emails carry no skill signal
    text = _ALLOWED.sub(" ", text)
    text = re.sub(r"(?<!\S)[./-]+(?!\S)", " ", text)   # stand-alone punctuation, e.g. bullet dashes
    text = re.sub(r"[./-]+(?=\s|$)", " ", text)          # sentence-ending full stops
    return _SPACES.sub(" ", text).strip()


def chunk_words(text: str, size: int = 200, overlap: int = 40) -> list[str]:
    """Split text into overlapping word windows.

    The embedding model only reads ~256 tokens at a time, so a full resume is
    embedded in pieces instead of being silently cut off after the first page.
    """
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)] if words else []
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, len(words) - overlap, step)]
