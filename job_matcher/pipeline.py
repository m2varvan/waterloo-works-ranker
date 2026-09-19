"""Orchestration layer: ties extraction, scoring and persistence together.

Keeping this separate from the Streamlit UI means the same logic can be
driven by a future FastAPI backend without changes to the core.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

from . import config, scoring
from .database import Database, Job
from .extraction import extract_text


@dataclass
class AddResult:
    filename: str
    status: str          # 'ok' | 'empty' | 'unsupported' | 'error' | 'duplicate_updated'
    message: str = ""
    final_score: Optional[float] = None
    was_update: bool = False


# --------------------------------------------------------------------------
# Lightweight metadata guesses (title / company) from the raw text.
# WaterlooWorks exports vary a lot, so these are best-effort only.
# --------------------------------------------------------------------------
def _guess_title(text: str, filename: str) -> str:
    for line in (text or "").splitlines():
        line = line.strip()
        if 3 < len(line) < 90 and not line.lower().startswith(("http", "www")):
            return line
    return os.path.splitext(os.path.basename(filename))[0]


def _guess_company(text: str) -> str:
    m = re.search(r"(?:company|organization|employer)\s*[:\-]\s*(.+)",
                  text or "", re.IGNORECASE)
    if m:
        return m.group(1).strip().splitlines()[0][:80]
    return ""


class Matcher:
    """High-level operations used by the UI."""

    def __init__(self, db: Optional[Database] = None,
                 weights: Optional[dict] = None,
                 use_embeddings: bool = True):
        self.db = db or Database()
        self.weights = weights or dict(config.DEFAULT_WEIGHTS)
        self.use_embeddings = use_embeddings and scoring.embeddings_available()
        self._resume_keywords: Optional[dict[str, float]] = None

    # -- resume / profile --------------------------------------------------
    def set_resume(self, resume_text: str, filename: str = "") -> None:
        """Persist the resume and its embedding, and cache its keywords."""
        embedding = (scoring.embed_text(resume_text)
                     if self.use_embeddings else None)
        self.db.save_profile(resume_text, embedding, filename)
        self._resume_keywords = scoring.build_resume_keywords(resume_text)

    def _resume_keyword_dict(self, resume_text: str) -> dict[str, float]:
        if self._resume_keywords is None:
            self._resume_keywords = scoring.build_resume_keywords(resume_text)
        return self._resume_keywords

    def has_resume(self) -> bool:
        return self.db.has_profile()

    # -- adding jobs -------------------------------------------------------
    def add_job_file(self, path: str,
                     display_name: Optional[str] = None) -> AddResult:
        """Extract, score and store a single job file (incremental add)."""
        filename = display_name or os.path.basename(path)
        profile = self.db.get_profile()
        if profile is None:
            return AddResult(filename, "error",
                             "Upload a resume before adding jobs.")

        extraction = extract_text(path)
        was_update = self.db.job_exists(filename)

        if extraction.status in ("unsupported", "error"):
            # Still record it so the user sees it flagged, but don't score.
            self.db.upsert_job(
                filename=filename, title=filename, company="",
                raw_text=extraction.text, embedding=None,
                tfidf_score=0, embedding_score=0, keyword_score=0,
                final_score=0, top_keywords="",
                status=extraction.status, note=extraction.message,
            )
            return AddResult(filename, extraction.status,
                             extraction.message, 0.0, was_update)

        if extraction.status == "empty":
            self.db.upsert_job(
                filename=filename, title=_guess_title(extraction.text, filename),
                company="", raw_text=extraction.text, embedding=None,
                tfidf_score=0, embedding_score=0, keyword_score=0,
                final_score=0, top_keywords="",
                status="empty", note=extraction.message,
            )
            return AddResult(filename, "empty", extraction.message, 0.0, was_update)

        # --- score it ---
        job_text = extraction.text
        job_embedding = (scoring.embed_text(job_text)
                         if self.use_embeddings else None)
        result = scoring.score_job(
            resume_text=profile.resume_text,
            job_text=job_text,
            resume_keywords=self._resume_keyword_dict(profile.resume_text),
            resume_embedding=profile.embedding,
            job_embedding=job_embedding,
            weights=self.weights,
            embeddings_enabled=self.use_embeddings,
        )

        self.db.upsert_job(
            filename=filename,
            title=_guess_title(job_text, filename),
            company=_guess_company(job_text),
            raw_text=job_text,
            embedding=job_embedding,
            tfidf_score=result.tfidf_score,
            embedding_score=result.embedding_score,
            keyword_score=result.keyword_score,
            final_score=result.final_score,
            top_keywords=result.keywords_display(),
            status="ok",
            note="",
        )
        status = "duplicate_updated" if was_update else "ok"
        return AddResult(filename, status, "", result.final_score, was_update)

    def add_folder(self, folder: str) -> list[AddResult]:
        from .extraction import list_supported_files
        results = []
        for path in list_supported_files(folder):
            results.append(self.add_job_file(path))
        return results

    # -- re-scoring --------------------------------------------------------
    def rescore_all(self) -> int:
        """Re-score every stored job against the current resume.

        Used when the resume changes. Reuses each job's stored raw text and
        embedding, so no re-extraction is needed. Returns count re-scored.
        """
        profile = self.db.get_profile()
        if profile is None:
            return 0
        keywords = self._resume_keyword_dict(profile.resume_text)
        count = 0
        for job in self.db.get_all_jobs(ranked=False):
            if job.status != "ok":
                continue
            job_embedding = self.db.get_job_embedding(job.id)
            result = scoring.score_job(
                resume_text=profile.resume_text,
                job_text=job.raw_text,
                resume_keywords=keywords,
                resume_embedding=profile.embedding,
                job_embedding=job_embedding,
                weights=self.weights,
                embeddings_enabled=self.use_embeddings,
            )
            self.db.update_job_scores(
                job.id,
                tfidf_score=result.tfidf_score,
                embedding_score=result.embedding_score,
                keyword_score=result.keyword_score,
                final_score=result.final_score,
                top_keywords=result.keywords_display(),
            )
            count += 1
        return count

    # -- reads -------------------------------------------------------------
    def ranked_jobs(self) -> list[Job]:
        return self.db.get_all_jobs(ranked=True)
