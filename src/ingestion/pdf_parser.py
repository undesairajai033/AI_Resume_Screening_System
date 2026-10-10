"""Extract plain text from resumes and job descriptions (PDF, DOCX or TXT)."""

import io
import os

import pdfplumber


def extract_text(source, filename: str | None = None) -> str:
    """Return the text of a PDF, DOCX or TXT file.

    `source` can be a file path, raw bytes, or a file-like object (e.g. a
    Streamlit or FastAPI upload). `filename` is used to detect the type when
    `source` is not a path.
    """
    name = (filename or (source if isinstance(source, str) else "")).lower()
    data = _read_bytes(source)
    ext = os.path.splitext(name)[1]

    if ext == ".pdf" or data[:4] == b"%PDF":
        return _pdf_text(data)
    if ext == ".docx":
        return _docx_text(data)
    return data.decode("utf-8", errors="ignore")


def _read_bytes(source) -> bytes:
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    if isinstance(source, str):
        with open(source, "rb") as f:
            return f.read()
    if hasattr(source, "seek"):
        source.seek(0)
    return source.read()


def _pdf_text(data: bytes) -> str:
    pages = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")  # scanned/image pages return None
    return "\n".join(pages)


def _docx_text(data: bytes) -> str:
    try:
        from docx import Document
    except ImportError as exc:
        raise RuntimeError("Reading .docx files needs: pip install python-docx") from exc
    doc = Document(io.BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs)
