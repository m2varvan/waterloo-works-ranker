"""Resume-to-Job Matcher - Streamlit UI.

Run with:  streamlit run app.py

Upload a resume, add job postings (single files or a whole folder), and get
a ranked list of best-fit jobs with a transparent per-job explanation.
"""

from __future__ import annotations

import os
import tempfile

import pandas as pd
import streamlit as st

from job_matcher import config, scoring
from job_matcher.database import Database
from job_matcher.extraction import (SUPPORTED_EXTENSIONS, extract_text,
                                     list_supported_files)
from job_matcher.pipeline import Matcher

st.set_page_config(page_title="Resume → Job Matcher", page_icon="📄", layout="wide")

UPLOAD_TYPES = [ext.lstrip(".") for ext in sorted(SUPPORTED_EXTENSIONS)]


# --------------------------------------------------------------------------
# Session-scoped resources
# --------------------------------------------------------------------------
@st.cache_resource
def get_db() -> Database:
    return Database()


def build_matcher() -> Matcher:
    weights = {
        "keyword": st.session_state.get("w_keyword", config.DEFAULT_WEIGHTS["keyword"]),
        "tfidf": st.session_state.get("w_tfidf", config.DEFAULT_WEIGHTS["tfidf"]),
        "embedding": st.session_state.get("w_embedding", config.DEFAULT_WEIGHTS["embedding"]),
    }
    use_emb = st.session_state.get("use_embeddings", True)
    return Matcher(db=get_db(), weights=weights, use_embeddings=use_emb)


def _save_upload_to_temp(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return tmp.name


# --------------------------------------------------------------------------
# Sidebar: scoring configuration
# --------------------------------------------------------------------------
def render_sidebar() -> None:
    st.sidebar.header("Scoring configuration")

    emb_ok = scoring.embeddings_available()
    st.session_state["use_embeddings"] = st.sidebar.checkbox(
        "Use local semantic embeddings (Layer 2)",
        value=emb_ok,
        disabled=not emb_ok,
        help="Uses sentence-transformers on-device. First run downloads "
             "the model (~90 MB).",
    )
    if not emb_ok:
        st.sidebar.warning(
            "sentence-transformers not installed. Scoring runs on the "
            "keyword + TF-IDF layers only. Install it to enable Layer 2."
        )

    st.sidebar.subheader("Layer weights")
    st.sidebar.caption("Weights are normalized automatically. "
                       "Default is a 50/50 blend of Layer 1 (keyword+TF-IDF) "
                       "and Layer 2 (embeddings).")
    st.session_state["w_keyword"] = st.sidebar.slider(
        "Keyword dictionary", 0.0, 1.0,
        config.DEFAULT_WEIGHTS["keyword"], 0.05)
    st.session_state["w_tfidf"] = st.sidebar.slider(
        "TF-IDF similarity", 0.0, 1.0,
        config.DEFAULT_WEIGHTS["tfidf"], 0.05)
    st.session_state["w_embedding"] = st.sidebar.slider(
        "Semantic embedding", 0.0, 1.0,
        config.DEFAULT_WEIGHTS["embedding"], 0.05,
        disabled=not st.session_state["use_embeddings"])

    with st.sidebar.expander("Optional: LLM reasoning (Layer 3)"):
        st.caption("Off by default. Paste your own API key for a qualitative "
                   "fit summary on individual jobs. No key = fully free.")
        st.session_state["llm_key"] = st.text_input(
            "OpenAI API key", type="password",
            value=st.session_state.get("llm_key", ""))

    db = get_db()
    st.sidebar.divider()
    st.sidebar.metric("Jobs stored", db.count_jobs())
    if st.sidebar.button("Clear all jobs", type="secondary"):
        db.clear_jobs()
        st.sidebar.success("Cleared all jobs.")
        st.rerun()


# --------------------------------------------------------------------------
# Resume upload section
# --------------------------------------------------------------------------
def render_resume_section() -> None:
    st.subheader("1. Your resume")
    db = get_db()
    profile = db.get_profile()

    if profile:
        st.success(f"Current resume: **{profile.filename or 'uploaded resume'}** "
                   f"({len(profile.resume_text)} chars extracted).")

    uploaded = st.file_uploader(
        "Upload resume (PDF, .tex, .docx, .txt)",
        type=UPLOAD_TYPES, key="resume_uploader")

    if uploaded is not None:
        if st.button("Set as my resume", type="primary"):
            path = _save_upload_to_temp(uploaded)
            try:
                result = extract_text(path)
            finally:
                os.unlink(path)

            if not result.ok:
                st.error(f"Could not use this resume: {result.message}")
                return

            matcher = build_matcher()
            with st.spinner("Saving resume and computing its embedding..."):
                matcher.set_resume(result.text, uploaded.name)

            st.success("Resume saved.")
            existing = db.count_jobs()
            if existing > 0:
                st.session_state["offer_rescore"] = True
            st.rerun()

    # Offer to re-score existing jobs after a resume change.
    if st.session_state.get("offer_rescore") and db.count_jobs() > 0:
        st.warning(f"Your resume changed and you have {db.count_jobs()} "
                   f"stored jobs. Re-score them against the new resume?")
        col1, col2 = st.columns(2)
        if col1.button("Re-score all jobs", type="primary"):
            matcher = build_matcher()
            with st.spinner("Re-scoring all jobs..."):
                n = matcher.rescore_all()
            st.session_state["offer_rescore"] = False
            st.success(f"Re-scored {n} jobs.")
            st.rerun()
        if col2.button("Skip for now"):
            st.session_state["offer_rescore"] = False
            st.rerun()


# --------------------------------------------------------------------------
# Job upload section
# --------------------------------------------------------------------------
def render_add_jobs_section() -> None:
    st.subheader("2. Add job postings")
    db = get_db()
    if not db.has_profile():
        st.info("Upload a resume first, then add jobs.")
        return

    tab_files, tab_folder = st.tabs(["Upload files", "Add a folder"])

    with tab_files:
        uploaded = st.file_uploader(
            "Upload one or more job postings",
            type=UPLOAD_TYPES, accept_multiple_files=True,
            key="job_uploader")
        if uploaded and st.button("Score and add", key="add_files_btn",
                                  type="primary"):
            matcher = build_matcher()
            results = []
            progress = st.progress(0.0)
            for i, f in enumerate(uploaded):
                path = _save_upload_to_temp(f)
                try:
                    results.append(matcher.add_job_file(path, display_name=f.name))
                finally:
                    os.unlink(path)
                progress.progress((i + 1) / len(uploaded))
            progress.empty()
            _report_add_results(results)
            st.rerun()

    with tab_folder:
        st.caption("Point to a local folder of job postings (great for a "
                   "batch of WaterlooWorks PDFs).")
        folder = st.text_input("Folder path", key="folder_path")
        if folder and st.button("Scan and add folder", key="add_folder_btn",
                                type="primary"):
            if not os.path.isdir(folder):
                st.error("That path is not a folder.")
            else:
                files = list_supported_files(folder)
                if not files:
                    st.warning("No supported files found in that folder.")
                else:
                    matcher = build_matcher()
                    results = []
                    progress = st.progress(0.0)
                    for i, path in enumerate(files):
                        results.append(matcher.add_job_file(path))
                        progress.progress((i + 1) / len(files))
                    progress.empty()
                    _report_add_results(results)
                    st.rerun()


def _report_add_results(results) -> None:
    added = sum(1 for r in results if r.status == "ok")
    updated = sum(1 for r in results if r.status == "duplicate_updated")
    empty = [r for r in results if r.status == "empty"]
    bad = [r for r in results if r.status in ("error", "unsupported")]

    if added:
        st.success(f"Added {added} new job(s).")
    if updated:
        st.info(f"Updated {updated} existing job(s) (duplicate detected).")
    if empty:
        st.warning("Flagged (no extractable text, likely scanned): "
                   + ", ".join(r.filename for r in empty))
    if bad:
        for r in bad:
            st.error(f"{r.filename}: {r.message}")


# --------------------------------------------------------------------------
# Ranked results
# --------------------------------------------------------------------------
def render_results_section() -> None:
    st.subheader("3. Ranked jobs")
    db = get_db()
    jobs = db.get_all_jobs(ranked=True)
    if not jobs:
        st.info("No jobs yet. Add some above.")
        return

    scored = [j for j in jobs if j.status == "ok"]
    flagged = [j for j in jobs if j.status != "ok"]

    if scored:
        table = pd.DataFrame([{
            "Rank": i + 1,
            "Score": j.final_score,
            "Title": j.title,
            "Company": j.company,
            "File": j.filename,
            "Keyword": j.keyword_score,
            "TF-IDF": j.tfidf_score,
            "Embedding": j.embedding_score,
        } for i, j in enumerate(scored)])

        st.dataframe(
            table,
            use_container_width=True, hide_index=True,
            column_config={
                "Score": st.column_config.ProgressColumn(
                    "Match score", min_value=0, max_value=100, format="%.1f"),
            },
        )

        st.download_button(
            "Download ranking (CSV)",
            table.to_csv(index=False).encode("utf-8"),
            "job_ranking.csv", "text/csv")

        st.markdown("#### Why each job scored the way it did")
        for i, j in enumerate(scored):
            with st.expander(
                f"#{i + 1}  •  {j.final_score:.1f}  •  {j.title}"
                + (f"  ({j.company})" if j.company else "")
            ):
                c1, c2, c3 = st.columns(3)
                c1.metric("Keyword", f"{j.keyword_score:.1f}")
                c2.metric("TF-IDF", f"{j.tfidf_score:.1f}")
                c3.metric("Embedding", f"{j.embedding_score:.1f}")
                if j.top_keywords:
                    st.markdown("**Top overlapping signals:** " + j.top_keywords)
                else:
                    st.caption("No strong keyword overlap detected.")

                _render_llm_button(j)

                if st.button("Remove this job", key=f"del_{j.id}"):
                    db.delete_job(j.id)
                    st.rerun()

    if flagged:
        st.markdown("#### ⚠️ Flagged jobs (not scored)")
        for j in flagged:
            st.warning(f"**{j.filename}** — {j.status}: {j.note}")
            if st.button("Remove", key=f"delflag_{j.id}"):
                db.delete_job(j.id)
                st.rerun()


def _render_llm_button(job) -> None:
    key = st.session_state.get("llm_key", "")
    if not key:
        return
    if st.button("Get LLM fit summary", key=f"llm_{job.id}"):
        db = get_db()
        profile = db.get_profile()
        with st.spinner("Asking the LLM..."):
            summary = scoring.llm_reasoning(profile.resume_text, job.raw_text, key)
        st.info(summary)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    st.title("📄 Resume → Job Matcher")
    st.caption("Rank WaterlooWorks postings against your resume so you can "
               "prioritize the best-fit applications. Runs fully on-device.")

    render_sidebar()
    render_resume_section()
    st.divider()
    render_add_jobs_section()
    st.divider()
    render_results_section()


if __name__ == "__main__":
    main()
