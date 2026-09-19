"""Tunable configuration for the job matcher.

Everything a user might want to adjust lives here so the rest of the
code stays generic. The Streamlit UI can override these at runtime.
"""

from __future__ import annotations

import os

# --- Paths -----------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "job_matcher.db")

# --- Embedding model -------------------------------------------------------
# Small, fast, downloads once (~90 MB) and then runs fully on-device.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Score blending --------------------------------------------------------
# Final score = keyword_weight * keyword + tfidf_weight * tfidf
#             + embedding_weight * embedding, then normalized to 0-100.
# The prompt asks for a configurable 50/50 blend of Layer 1 (keyword+tfidf)
# and Layer 2 (embeddings). Layer 1 is itself split between the transparent
# keyword dictionary and the TF-IDF cosine similarity.
DEFAULT_WEIGHTS = {
    "keyword": 0.25,    # weighted keyword dictionary  (part of Layer 1)
    "tfidf": 0.25,      # TF-IDF cosine similarity      (part of Layer 1)
    "embedding": 0.50,  # local sentence-transformer   (Layer 2)
}

# --- Keyword extraction ----------------------------------------------------
# How many of the resume's own top terms become the weighted keyword
# dictionary used by Layer 1.
TOP_RESUME_KEYWORDS = 40

# Number of overlapping keywords surfaced in the per-job explanation.
TOP_EXPLANATION_KEYWORDS = 8

# --- Base domain signals (from the reference implementation) ---------------
# These are profile-agnostic domain buckets. Positive buckets reward a good
# fit; negative buckets penalize a poor fit. They are BLENDED with keywords
# auto-extracted from whatever resume the user uploads, so the app works for
# any user while still capturing broad domain fit out of the box.
POSITIVE_SIGNALS = [
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

NEGATIVE_SIGNALS = [
    (["cybersecurity", "penetration testing", "network security", "wi-fi",
      "wifi", "embedded system", "firmware"], -20, "[penalty] Cybersec/hardware"),
    (["equity research", "financial modeling", "investment banking",
      "capital markets", "bloomberg"], -15, "[penalty] Finance heavy"),
    (["lean manufacturing", "six sigma", "autocad", "solidworks",
      "procurement", "logistics"], -10, "[penalty] Manufacturing/ops"),
    (["sales", "marketing", "business development"], -10, "[penalty] Sales/marketing"),
]
