"""Recruiter dashboard.  Run from the project root:  streamlit run dashboard/app.py"""

import os
import sys

import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.pdf_parser import extract_text  # noqa: E402
from src.models.ranking_model import rank_resumes  # noqa: E402

SAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "samples")

st.set_page_config(page_title="AI Resume Screening", page_icon="📄", layout="wide")
st.title("📄 AI Resume Screening System")
st.caption("Ranks resumes against a job description by meaning (sentence embeddings) and required skills.")


@st.cache_resource(show_spinner="Loading the language model (first run downloads ~90 MB)...")
def warm_up():
    from src.feature_engineering.embeddings import get_embedding
    get_embedding("warm up")
    return True


def load_samples():
    jd_path = os.path.join(SAMPLES, "job_description.txt")
    jd = open(jd_path, encoding="utf-8").read() if os.path.exists(jd_path) else ""
    resumes = {}
    folder = os.path.join(SAMPLES, "resumes")
    if os.path.isdir(folder):
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith((".pdf", ".docx", ".txt")):
                resumes[f] = extract_text(os.path.join(folder, f))
    return jd, resumes


# ── Inputs ────────────────────────────────────────────────────────────────────
use_samples = st.sidebar.toggle("Use sample data", help="Fictional resumes and a sample job description")
semantic_weight = st.sidebar.slider(
    "Weight on meaning vs. skills", 0.0, 1.0, 0.7, 0.05,
    help="1.0 = only semantic similarity, 0.0 = only required-skill coverage",
)

sample_jd, sample_resumes = load_samples() if use_samples else ("", {})

jd_text = st.text_area("Job description", value=sample_jd, height=220,
                       placeholder="Paste the full job description here...")
uploads = st.file_uploader("Resumes (PDF, DOCX or TXT)", type=["pdf", "docx", "txt"],
                           accept_multiple_files=True)

resumes = dict(sample_resumes)
for up in uploads or []:
    try:
        resumes[up.name] = extract_text(up, filename=up.name)
    except Exception:
        st.warning(f"Couldn't read {up.name}; skipped.")

empty = [name for name, text in resumes.items() if len(text.strip()) < 50]
for name in empty:
    st.warning(f"{name} has almost no readable text (scanned image?). Skipped.")
    resumes.pop(name)

# ── Ranking ───────────────────────────────────────────────────────────────────
if st.button("Rank candidates", type="primary", disabled=not (jd_text.strip() and resumes)):
    warm_up()
    with st.spinner(f"Scoring {len(resumes)} resumes..."):
        ranked = rank_resumes(jd_text, resumes, semantic_weight=semantic_weight)
    st.session_state["ranked"] = ranked

ranked = st.session_state.get("ranked")
if ranked is not None and not ranked.empty:
    st.subheader("Ranked shortlist")
    top = ranked.iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Resumes screened", len(ranked))
    c2.metric("Best match", top["candidate"])
    c3.metric("Best score", f"{top['final_score']:.2f}")

    table = ranked.assign(
        matched_skills=ranked["matched_skills"].apply(", ".join),
        missing_skills=ranked["missing_skills"].apply(", ".join),
    )
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "rank": st.column_config.NumberColumn("Rank", width="small"),
            "candidate": "Candidate",
            "final_score": st.column_config.ProgressColumn("Score", min_value=0, max_value=1, format="%.2f"),
            "semantic_score": st.column_config.NumberColumn("Meaning", format="%.2f"),
            "skill_coverage": st.column_config.NumberColumn("Skills covered", format="percent"),
            "matched_skills": "Has",
            "missing_skills": "Missing",
        },
    )

    chart = ranked.melt(id_vars="candidate", value_vars=["semantic_score", "skill_coverage"],
                        var_name="part", value_name="score")
    chart["part"] = chart["part"].map({"semantic_score": "Meaning match", "skill_coverage": "Skills covered"})
    st.plotly_chart(px.bar(chart, x="score", y="candidate", color="part", barmode="group", orientation="h",
                           labels={"candidate": "", "score": "", "part": ""}).update_yaxes(autorange="reversed"),
                    width="stretch")

    st.download_button("Download results (CSV)", table.to_csv(index=False), "ranked_candidates.csv", "text/csv")
elif not resumes:
    st.info("Upload resumes, or turn on **Use sample data** in the sidebar to try it out.")
