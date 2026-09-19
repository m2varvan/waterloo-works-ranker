"""Scoring engine.

Three transparent layers, blended into a final 0-100 score:

    Layer 1a: Weighted keyword dictionary.
        - Broad domain buckets from config (profile-agnostic starting point).
        - PLUS keywords auto-extracted from the uploaded resume via TF-IDF,
          weighted by how prominent each term is in the resume. This is what
          makes the app generic: it learns what matters from *your* resume.
    Layer 1b: TF-IDF cosine similarity between resume and job text.
    Layer 2:  Local sentence-transformer embedding cosine similarity.

    Final = w_kw * keyword + w_tfidf * tfidf + w_emb * embedding  -> 0-100.

The embedding model is loaded lazily and cached so importing this module
(and running tests on the other layers) never forces a model download.

Optional Layer 3 (LLM reasoning) is provided but OFF by default and only
runs if the user supplies their own API key.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import numpy as np

from . import config

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+#.\-]{1,}")

# Very small stopword list; scikit-learn's TF-IDF also strips English stops.
_STOPWORDS = {
    "the", "and", "for", "with", "you", "your", "our", "will", "are", "this",
    "that", "have", "has", "from", "job", "work", "team", "role", "including",
    "ability", "experience", "years", "year", "using", "use", "used", "such",
    "who", "what", "when", "who", "why", "how", "all", "any", "may", "can",
    "must", "should", "would", "into", "out", "not", "but", "per", "etc",
    "responsibilities", "requirements", "qualifications", "opportunity",
    "company", "position", "candidate", "candidates", "waterloo", "coop",
}


# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------
@dataclass
class ScoreResult:
    final_score: float          # 0-100
    tfidf_score: float          # 0-100 component
    embedding_score: float      # 0-100 component
    keyword_score: float        # 0-100 component
    top_keywords: list[str] = field(default_factory=list)
    domain_signals: list[str] = field(default_factory=list)

    def keywords_display(self) -> str:
        parts = list(self.top_keywords)
        if self.domain_signals:
            parts += [s for s in self.domain_signals if s not in parts]
        return ", ".join(parts)


# --------------------------------------------------------------------------
# Tokenisation helpers
# --------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    tokens = [t.lower() for t in _TOKEN_RE.findall(text or "")]
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 2]


# --------------------------------------------------------------------------
# Embedding model (lazy singleton)
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _get_model(model_name: str):
    """Load and cache the sentence-transformer model.

    Imported lazily so the rest of the app works even before the (large)
    dependency is installed or the model is downloaded.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_text(text: str, model_name: str = config.EMBEDDING_MODEL
               ) -> Optional[np.ndarray]:
    """Return a normalized embedding vector, or None if text is empty."""
    if not text or not text.strip():
        return None
    model = _get_model(model_name)
    vec = model.encode(text, normalize_embeddings=True)
    return np.asarray(vec, dtype=np.float32)


def embeddings_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------
# Layer 1a: weighted keyword dictionary derived from the resume
# --------------------------------------------------------------------------
def build_resume_keywords(resume_text: str,
                          top_n: int = config.TOP_RESUME_KEYWORDS
                          ) -> dict[str, float]:
    """Extract the resume's own prominent terms as a weighted dictionary.

    Uses TF-IDF over the resume treated as a single document; weights are
    normalized to sum to 1 so the keyword score is comparable across
    resumes. Returns {term: weight}.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        # Fallback: plain frequency counts.
        return _frequency_keywords(resume_text, top_n)

    if not resume_text.strip():
        return {}

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        token_pattern=r"[a-zA-Z][a-zA-Z0-9+#.\-]{1,}",
        max_features=400,
    )
    try:
        matrix = vectorizer.fit_transform([resume_text.lower()])
    except ValueError:
        return _frequency_keywords(resume_text, top_n)

    scores = matrix.toarray()[0]
    terms = vectorizer.get_feature_names_out()
    ranked = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)
    ranked = [(t, s) for t, s in ranked
              if s > 0 and t not in _STOPWORDS][:top_n]
    total = sum(s for _, s in ranked) or 1.0
    return {t: s / total for t, s in ranked}


def _frequency_keywords(text: str, top_n: int) -> dict[str, float]:
    counts: dict[str, int] = {}
    for tok in tokenize(text):
        counts[tok] = counts.get(tok, 0) + 1
    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    total = sum(c for _, c in ranked) or 1.0
    return {t: c / total for t, c in ranked}


def keyword_score(job_text: str, resume_keywords: dict[str, float]
                  ) -> tuple[float, list[str], list[str]]:
    """Score a job on keyword overlap with the resume + domain signals.

    Returns (score_0_100, matched_resume_terms, domain_signal_labels).
    """
    text = (job_text or "").lower()

    # -- resume-derived keyword overlap --
    matched: list[tuple[str, float]] = []
    overlap_weight = 0.0
    for term, weight in resume_keywords.items():
        if term in text:
            overlap_weight += weight
            matched.append((term, weight))
    matched.sort(key=lambda x: x[1], reverse=True)
    matched_terms = [t for t, _ in matched[:config.TOP_EXPLANATION_KEYWORDS]]

    # Since weights sum to ~1, overlap_weight is already in [0, 1].
    resume_component = min(overlap_weight, 1.0)

    # -- broad domain signal buckets (from the reference implementation) --
    domain_raw = 0
    domain_labels: list[str] = []
    for keywords, weight, label in config.POSITIVE_SIGNALS + config.NEGATIVE_SIGNALS:
        if any(k in text for k in keywords):
            domain_raw += weight
            domain_labels.append(label)

    # Map the domain raw score into [0, 1]. The positive buckets sum to ~230,
    # but any realistic job hits only a few. Squash with a soft curve so a
    # handful of strong matches already yields a high component.
    domain_component = 1.0 / (1.0 + math.exp(-domain_raw / 40.0))

    # Blend resume-specific (60%) and generic domain fit (40%).
    combined = 0.6 * resume_component + 0.4 * domain_component
    return round(combined * 100.0, 2), matched_terms, domain_labels


# --------------------------------------------------------------------------
# Layer 1b: TF-IDF cosine similarity
# --------------------------------------------------------------------------
def tfidf_score(resume_text: str, job_text: str) -> float:
    """Cosine similarity of resume vs job TF-IDF vectors, scaled 0-100."""
    if not resume_text.strip() or not job_text.strip():
        return 0.0
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        return 0.0

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        token_pattern=r"[a-zA-Z][a-zA-Z0-9+#.\-]{1,}",
    )
    try:
        matrix = vectorizer.fit_transform([resume_text.lower(), job_text.lower()])
    except ValueError:
        return 0.0
    sim = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
    return round(float(sim) * 100.0, 2)


# --------------------------------------------------------------------------
# Layer 2: embedding cosine similarity
# --------------------------------------------------------------------------
def embedding_score(resume_embedding: Optional[np.ndarray],
                    job_embedding: Optional[np.ndarray]) -> float:
    """Cosine similarity of two (normalized) embeddings, scaled 0-100.

    Embeddings from ``embed_text`` are already L2-normalized, so a dot
    product is the cosine similarity. We clamp the [-1, 1] range to [0, 1]
    before scaling because negative similarity is meaningless for ranking.
    """
    if resume_embedding is None or job_embedding is None:
        return 0.0
    a = np.asarray(resume_embedding, dtype=np.float32)
    b = np.asarray(job_embedding, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    sim = float(np.dot(a, b) / denom)
    sim = max(0.0, sim)  # clamp negatives to 0
    return round(sim * 100.0, 2)


# --------------------------------------------------------------------------
# Blended final score
# --------------------------------------------------------------------------
def combine_scores(keyword: float, tfidf: float, embedding: float,
                   weights: Optional[dict[str, float]] = None,
                   embeddings_enabled: bool = True) -> float:
    """Weighted blend of the three components, normalized to 0-100.

    If embeddings are disabled/unavailable, its weight is redistributed
    across the remaining layers so the final score still spans 0-100.
    """
    w = dict(weights or config.DEFAULT_WEIGHTS)
    if not embeddings_enabled:
        w["embedding"] = 0.0

    total_w = sum(w.values()) or 1.0
    final = (
        w["keyword"] * keyword
        + w["tfidf"] * tfidf
        + w["embedding"] * embedding
    ) / total_w
    return round(max(0.0, min(100.0, final)), 2)


def score_job(*, resume_text: str, job_text: str,
              resume_keywords: dict[str, float],
              resume_embedding: Optional[np.ndarray] = None,
              job_embedding: Optional[np.ndarray] = None,
              weights: Optional[dict[str, float]] = None,
              embeddings_enabled: bool = True) -> ScoreResult:
    """Run all layers for a single job and return a ``ScoreResult``."""
    kw, matched_terms, domain_labels = keyword_score(job_text, resume_keywords)
    tf = tfidf_score(resume_text, job_text)
    emb = embedding_score(resume_embedding, job_embedding) if embeddings_enabled else 0.0

    final = combine_scores(kw, tf, emb, weights, embeddings_enabled)
    return ScoreResult(
        final_score=final,
        tfidf_score=tf,
        embedding_score=emb,
        keyword_score=kw,
        top_keywords=matched_terms,
        domain_signals=domain_labels,
    )


# --------------------------------------------------------------------------
# Optional Layer 3: LLM qualitative pass (off by default)
# --------------------------------------------------------------------------
def llm_reasoning(resume_text: str, job_text: str, api_key: str,
                  model: str = "gpt-4o-mini") -> str:
    """Optional qualitative fit assessment using the user's own API key.

    Returns a short natural-language explanation. Any failure is returned
    as a message string rather than raised, so the free layers keep working.
    This function is never called unless the user explicitly supplies a key.
    """
    if not api_key:
        return "No API key provided; LLM layer is disabled."
    try:
        from openai import OpenAI
    except ImportError:
        return ("openai package not installed. Run `pip install openai` "
                "to enable the optional LLM layer.")

    prompt = (
        "You are a co-op recruiter. In 2-3 sentences, assess how well this "
        "candidate's resume fits the job posting and note the biggest gap.\n\n"
        f"RESUME:\n{resume_text[:4000]}\n\nJOB POSTING:\n{job_text[:4000]}"
    )
    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=200,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:  # noqa: BLE001
        return f"LLM call failed: {exc}"
