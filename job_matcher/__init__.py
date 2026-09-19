"""Resume-to-Job Matcher core package.

Modules:
    extraction  - text extraction from PDF/.tex/.docx/.txt
    database    - SQLite persistence layer
    scoring     - keyword + TF-IDF + embedding scoring
    config      - tunable defaults (weights, model name)
"""

__all__ = ["extraction", "database", "scoring", "config"]
