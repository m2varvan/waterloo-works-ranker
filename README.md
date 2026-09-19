# WaterlooWorks Ranker — Resume → Job Matcher

A local web app that ranks job postings against your resume so you can decide
which ones to apply to. Built for University of Waterloo WaterlooWorks, where
there are often far more postings than the number of applications you're
allowed, so prioritizing the best-fit jobs matters.

Everything runs on-device. No paid API is required.

## Features

- Upload a resume (PDF, `.tex`, `.docx`, or `.txt`) as your baseline profile.
- Add job postings one at a time, several at once, or by pointing at a whole
  folder of files.
- Each job is scored against your resume and inserted into a ranked list.
- Ranked table from best match to worst, with a per-job explanation of the
  top overlapping skills/terms that drove the score.
- Incremental adds: a new job is scored and slotted into the existing list
  without reprocessing everything else. The list re-sorts automatically.
- Data persists between sessions in a local SQLite database.
- Edge-case handling: duplicate detection, empty/scanned-PDF flagging, and an
  offer to re-score all jobs when you change your resume.

## Quick start

```bash
# 1. (recommended) create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. run the app
streamlit run app.py
```

The app opens in your browser. Upload your resume, add job postings, and read
the ranking. The first time embeddings run, the `all-MiniLM-L6-v2` model
(~90 MB) downloads once and is then cached locally.

> Note: `sentence-transformers` (Layer 2) is optional. If it isn't installed,
> the app automatically falls back to the keyword + TF-IDF layers and still
> produces a full 0–100 ranking.

## How the scoring works

Each job gets a final score from **0 to 100**, blended from three transparent
layers:

| Layer | What it measures | Default weight |
|-------|------------------|----------------|
| **1a. Keyword dictionary** | Overlap between the job text and a weighted dictionary of your resume's own top terms, blended with broad domain buckets (AI/ML, data, cloud, etc.). | 0.25 |
| **1b. TF-IDF cosine** | Cosine similarity between the resume and the job description as TF-IDF vectors (scikit-learn). | 0.25 |
| **2. Semantic embedding** | Cosine similarity of local `sentence-transformers` embeddings of the resume and the job. Captures meaning beyond exact word matches. | 0.50 |

```
final = (w_keyword·keyword + w_tfidf·tfidf + w_embedding·embedding)
        / (w_keyword + w_tfidf + w_embedding)      →  clamped to 0–100
```

The default is a **50/50 blend of Layer 1 (keyword + TF-IDF) and Layer 2
(embeddings)**. Weights are normalized automatically, so you can move the
sliders freely without them needing to sum to 1. If embeddings are disabled,
that weight is redistributed across the remaining layers.

The **per-job explanation** lists the resume terms and domain signals that
overlapped with the posting, so the ranking is never a black box.

### Optional Layer 3 — LLM reasoning (off by default)

In the sidebar you can paste your own OpenAI API key to get a short
qualitative fit summary on individual jobs. This is entirely optional; with no
key the app is fully free and never makes a network call for scoring. Enabling
it requires `pip install openai`.

## Tuning the keyword weights

There are two ways to tune what matters:

1. **Automatic (per resume).** Layer 1a extracts your resume's most prominent
   terms with TF-IDF and weights job matches by how prominent each term is in
   *your* resume. Just upload a different resume and the dictionary changes
   with it — no code edits needed.

2. **Manual (domain buckets).** The broad domain signals live in
   [`job_matcher/config.py`](job_matcher/config.py) as `POSITIVE_SIGNALS` and
   `NEGATIVE_SIGNALS`. Each entry is `(keywords, weight, label)`. Positive
   weights reward a good fit; negative weights penalize a poor fit. Edit these
   lists to reflect your own strengths and the domains you want to avoid. The
   `label` is what shows up in the per-job explanation.

You can also change the blend weights and the number of keywords surfaced by
editing `DEFAULT_WEIGHTS`, `TOP_RESUME_KEYWORDS`, and
`TOP_EXPLANATION_KEYWORDS` in `config.py`, or by moving the sliders in the app.

## Project structure

```
waterloo-works-ranker/
├── app.py                     # Streamlit UI (thin; delegates to the package)
├── requirements.txt
├── README.md
└── job_matcher/               # Core logic — UI-agnostic, reusable
    ├── config.py              # Tunable weights, model name, keyword buckets
    ├── extraction.py          # PDF / .tex / .docx / .txt text extraction
    ├── database.py            # SQLite schema + data-access layer
    ├── scoring.py             # Keyword, TF-IDF, embedding, blended scoring
    └── pipeline.py            # Orchestration: extract → score → persist
```

The scoring and data logic live in `job_matcher/`, separate from the UI, so
the Streamlit front end can later be swapped for React + FastAPI without
rewriting the core.

## Data model

- **`profile`** — a single row holding the resume text, its cached embedding,
  the source filename, and a timestamp.
- **`jobs`** — one row per posting: `id`, `filename` (unique, used for
  duplicate detection), `title`, `company`, `raw_text`, cached `embedding`,
  the three score components (`keyword_score`, `tfidf_score`,
  `embedding_score`), the blended `final_score`, `top_keywords`, extraction
  `status`/`note`, and `date_added`.

The database file (`job_matcher.db`) is created next to the app on first run
and is git-ignored.

## Edge cases handled

- **Duplicate jobs** — re-adding a file with the same name updates the existing
  row instead of creating a second one.
- **Empty / scanned PDFs** — if extraction yields little or no text, the job is
  flagged in the UI rather than silently scored as zero. (OCR is out of scope.)
- **Unsupported / unreadable files** — recorded and flagged with the error, not
  scored.
- **Resume changes** — after uploading a new resume, the app offers to re-score
  every stored job against it (reusing cached text/embeddings, so no
  re-extraction).
