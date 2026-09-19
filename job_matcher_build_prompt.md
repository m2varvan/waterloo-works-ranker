# Project Build Prompt: Resume to Job Matcher

Build a local web application that ranks job postings against my resume so I can decide which ones to apply to. This is for University of Waterloo WaterlooWorks applications, where there are often many postings but a limited number of applications allowed, so I need to prioritize the best-fit jobs.

## Core requirements

1. **Upload a resume** (PDF, .tex, .docx, or .txt) that serves as the baseline profile.
2. **Add job postings** either by uploading a whole folder of PDFs at once, or by adding a single PDF/file at a time.
3. **Process each job**: extract text, score it against the resume, and store it.
4. **Display a ranked list** of jobs from best match to worst, with the match score and a short explanation of why each job scored the way it did (which skills/terms overlapped).
5. **Incremental updates**: when a new job is added, it is scored and inserted into the existing ranked list without reprocessing the jobs already there. The list re-sorts automatically.
6. **Persist data** between sessions so I don't lose my ranked list when I close the app.

## Scoring approach (no paid API required)

Implement scoring in two free layers and combine them into a final score:

- **Layer 1: Keyword and TF-IDF similarity.** Extract text from the resume and each job. Compute a TF-IDF cosine similarity between the resume and each job description. Also maintain a configurable weighted keyword dictionary (see the reference implementation below) so I can tune what matters to me.
- **Layer 2: Local semantic embeddings.** Use the `sentence-transformers` library with a small local model (e.g. `all-MiniLM-L6-v2`) to embed the resume and each job, then rank by cosine similarity of the embeddings. Runs fully on-device, no API cost after a one-time model download.
- **Final score**: a weighted blend of Layer 1 and Layer 2 (make the weights configurable, default 50/50). Normalize to a 0 to 100 scale.
- **Optional Layer 3 (stretch goal, off by default):** allow the user to paste their own LLM API key for a qualitative reasoning pass. If no key is provided, the app works fully on the free layers.

## Explanation output

For each job, show the top overlapping keywords/skills that drove the score, so the ranking is transparent and not a black box.

## Tech stack

- **Backend/logic:** Python. Use `pdfplumber` for PDF text extraction, `scikit-learn` for TF-IDF and cosine similarity, `sentence-transformers` for embeddings, and `SQLite` for persistence.
- **Frontend:** Start with **Streamlit** for a fast working v1 (single file, easy folder upload, ranked table display). Structure the scoring and data logic in separate modules so the front end can later be swapped for React + FastAPI without rewriting the core.
- **Data model:** a `jobs` table (id, filename, title, company, raw_text, tfidf_score, embedding_score, final_score, top_keywords, date_added) and a `profile` table (resume text and its embedding).

## Handle these edge cases

- Duplicate job detection (same file added twice should update, not duplicate).
- PDFs where text extraction fails or returns empty (flag them clearly rather than scoring them as zero silently).
- Resume changes: if I upload a new resume, offer to re-score all existing jobs against it.
- Non-text/scanned PDFs: detect and warn (OCR is out of scope for v1).

## Deliverables

1. A working Streamlit app I can run with one command.
2. A `requirements.txt`.
3. A README explaining how to run it, how the scoring works, and how to tune the keyword weights.
4. Clean, modular code: separate files for PDF extraction, scoring, database, and the UI.

## Build order

1. PDF/resume text extraction module (test it on sample files first).
2. SQLite schema and data-access layer.
3. TF-IDF scoring, then embedding scoring, then the blended final score.
4. Streamlit UI: resume upload, folder/file upload, ranked table, per-job explanation.
5. Incremental add and re-sort.
6. Edge case handling and README.

Start by scaffolding the project structure and the extraction module, then confirm extraction works on a couple of sample PDFs before moving on.

---

## Reference implementation for Layer 1 (weighted keyword scoring)

This is a working keyword scoring function tuned to my profile (AI/ML, data science, software engineering, cloud, Microsoft Power Platform). Use it as the starting point for Layer 1's keyword dictionary. The weights and penalties reflect my background: strong in AI/RAG/LLM, ML, Python, cloud, and analytics; weaker fit for cybersecurity, hardware, heavy finance, and manufacturing operations.

```python
import pdfplumber
import os


def extract_text(pdf_path):
    """Extract lowercased text from a job posting PDF."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + " "
    except Exception:
        return ""
    return text.lower()


def score_job(text):
    """
    Weighted keyword scoring tuned to an AI/ML + data science + software profile.
    Returns (score, list_of_matched_signals).
    Positive weights reward matching skills; negative weights penalize poor-fit domains.
    """
    score = 0
    reasons = []

    positive_signals = [
        (["rag", "retrieval-augmented", "llm", "large language model",
          "generative ai", "genai", "agentic"], 30, "RAG/LLM/GenAI/Agentic"),
        (["machine learning", "ml model", "scikit", "xgboost", "deep learning",
          "neural network", "pytorch", "tensorflow"], 25, "ML modeling"),
        (["python", "pandas", "numpy"], 20, "Python"),
        (["azure", "gcp", "google cloud", "cloud platform"], 15, "Azure/GCP"),
        (["data pipeline", "data engineering", "etl", "pyspark", "spark",
          "duckdb", "parquet"], 15, "Data pipelines"),
        (["nlp", "natural language", "embeddings", "vector",
          "semantic search"], 20, "NLP/embeddings"),
        (["sql", "database", "mysql", "postgresql"], 10, "SQL"),
        (["power bi", "power automate", "power apps", "microsoft fabric",
          "microsoft 365"], 15, "MS Power Platform"),
        (["react", "node.js", "nodejs", "express", "javascript", "typescript",
          "full-stack", "fullstack", "frontend", "front-end"], 10, "Full-stack/JS"),
        (["firebase", "rest api"], 5, "API/Firebase"),
        (["data science", "data scientist", "data analyst",
          "analytics"], 15, "Data Science/Analytics"),
        (["co-op", "coop", "intern", "undergrad"], 5, "Co-op level"),
        (["copilot studio", "copilot", "ai foundry",
          "agent framework"], 10, "Copilot/AI Foundry"),
        (["feature engineering", "model evaluation", "model training",
          "regression", "classification"], 15, "Feature eng/modeling"),
        (["software engineer", "software developer", "software development",
          "backend"], 10, "Software engineering"),
    ]

    negative_signals = [
        (["cybersecurity", "penetration testing", "network security", "wi-fi",
          "wifi", "embedded system", "firmware"], -20, "[penalty] Cybersec/hardware"),
        (["equity research", "financial modeling", "investment banking",
          "capital markets", "bloomberg"], -15, "[penalty] Finance heavy"),
        (["lean manufacturing", "six sigma", "autocad", "solidworks",
          "procurement", "logistics"], -10, "[penalty] Manufacturing/ops"),
        (["sales", "marketing", "business development"], -10, "[penalty] Sales/marketing"),
    ]

    for keywords, weight, label in positive_signals + negative_signals:
        if any(k in text for k in keywords):
            score += weight
            reasons.append(label)

    return score, reasons


def rank_folder(folder_path):
    """Score every PDF in a folder and return a ranked list."""
    results = []
    for fname in sorted(os.listdir(folder_path)):
        if not fname.endswith(".pdf"):
            continue
        text = extract_text(os.path.join(folder_path, fname))
        score, reasons = score_job(text)
        results.append({"file": fname, "score": score, "signals": reasons})
    return sorted(results, key=lambda x: x["score"], reverse=True)
```

**Notes for adapting this:**
- The keyword weights above are hardcoded to my profile. In the app, generate or tune these from the uploaded resume so it works for any user, not just me. A simple approach: extract the resume's own keywords and weight job matches by how prominent each term is in the resume.
- This keyword layer is fast and transparent but shallow. Combine it with the TF-IDF and embedding layers for a more robust final score.
- The `reasons` list is what powers the per-job explanation in the UI.
