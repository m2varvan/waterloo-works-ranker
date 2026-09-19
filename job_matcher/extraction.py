"""Text extraction for resumes and job postings.

Supports PDF, LaTeX (.tex), Word (.docx) and plain text (.txt).

Every extractor returns an ``ExtractionResult`` so the caller can tell the
difference between "no text at all" (a scanned / image-only PDF) and a
genuine failure, rather than silently scoring an empty document as zero.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

# Minimum number of characters before we consider extraction "successful".
# Scanned PDFs typically yield a handful of stray characters or nothing.
MIN_MEANINGFUL_CHARS = 30

SUPPORTED_EXTENSIONS = {".pdf", ".tex", ".docx", ".txt"}


@dataclass
class ExtractionResult:
    """Outcome of extracting text from a single file."""

    text: str = ""
    ok: bool = False
    # 'ok' | 'empty' | 'unsupported' | 'error'
    status: str = "error"
    message: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        return len(self.text.strip())


# --------------------------------------------------------------------------
# Individual format extractors
# --------------------------------------------------------------------------
def _extract_pdf(path: str) -> tuple[str, str]:
    """Return (text, error_message). error_message is '' on success."""
    try:
        import pdfplumber
    except ImportError:
        return "", "pdfplumber is not installed (pip install pdfplumber)"

    text_parts: list[str] = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
    except Exception as exc:  # noqa: BLE001 - surface any parser failure
        return "", f"Failed to read PDF: {exc}"
    return " ".join(text_parts), ""


def _strip_latex(raw: str) -> str:
    """Best-effort removal of LaTeX markup so scoring sees readable prose."""
    # Drop comments.
    raw = re.sub(r"(?<!\\)%.*", "", raw)
    # Remove common formatting commands but keep their argument text.
    raw = re.sub(r"\\(section|subsection|textbf|textit|emph|item|href"
                 r"|underline|textsc|texttt)\*?\{", " ", raw)
    # Drop commands that carry no useful text (e.g. \vspace{1em}).
    raw = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^{}]*\})?", " ", raw)
    # Remaining braces / math delimiters.
    raw = raw.replace("{", " ").replace("}", " ").replace("$", " ")
    raw = raw.replace("\\&", "&").replace("~", " ").replace("\\", " ")
    return re.sub(r"\s+", " ", raw).strip()


def _extract_tex(path: str) -> tuple[str, str]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            raw = fh.read()
    except Exception as exc:  # noqa: BLE001
        return "", f"Failed to read .tex: {exc}"
    return _strip_latex(raw), ""


def _extract_docx(path: str) -> tuple[str, str]:
    try:
        import docx  # python-docx
    except ImportError:
        return "", "python-docx is not installed (pip install python-docx)"
    try:
        document = docx.Document(path)
    except Exception as exc:  # noqa: BLE001
        return "", f"Failed to read .docx: {exc}"

    parts = [p.text for p in document.paragraphs if p.text]
    # Also pull text out of tables, which resumes frequently use for layout.
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text:
                    parts.append(cell.text)
    return "\n".join(parts), ""


def _extract_txt(path: str) -> tuple[str, str]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            return fh.read(), ""
    except Exception as exc:  # noqa: BLE001
        return "", f"Failed to read .txt: {exc}"


_EXTRACTORS = {
    ".pdf": _extract_pdf,
    ".tex": _extract_tex,
    ".docx": _extract_docx,
    ".txt": _extract_txt,
}


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def extract_text(path: str) -> ExtractionResult:
    """Extract text from a file, returning a rich ``ExtractionResult``.

    The result distinguishes a successful read, an empty/scanned document,
    an unsupported format, and a hard error so the UI can react
    appropriately instead of treating every failure as a zero score.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext not in _EXTRACTORS:
        return ExtractionResult(
            status="unsupported",
            message=f"Unsupported file type '{ext}'. "
                    f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}.",
        )

    text, error = _EXTRACTORS[ext](path)
    if error:
        return ExtractionResult(status="error", message=error)

    cleaned = text.strip()
    if len(cleaned) < MIN_MEANINGFUL_CHARS:
        warn = ("Little or no extractable text found. "
                "This is likely a scanned / image-only document "
                "(OCR is out of scope).") if ext == ".pdf" else \
               "Little or no extractable text found in this file."
        return ExtractionResult(
            text=cleaned,
            ok=False,
            status="empty",
            message=warn,
            warnings=[warn],
        )

    return ExtractionResult(text=cleaned, ok=True, status="ok")


def extract_text_str(path: str) -> str:
    """Convenience wrapper returning just the (possibly empty) text."""
    return extract_text(path).text


def is_supported(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in SUPPORTED_EXTENSIONS


def list_supported_files(folder: str) -> list[str]:
    """Return sorted absolute paths of supported files inside ``folder``."""
    out: list[str] = []
    for name in sorted(os.listdir(folder)):
        full = os.path.join(folder, name)
        if os.path.isfile(full) and is_supported(name):
            out.append(full)
    return out
