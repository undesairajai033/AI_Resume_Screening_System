"""Rank resumes against a job description.

final_score = SEMANTIC_WEIGHT * semantic similarity  (sentence embeddings)
            + SKILL_WEIGHT    * skill coverage       (share of the job's skills found)

Semantic similarity catches meaning ("built REST services in FastAPI" ≈ "Python
backend API development"); skill coverage keeps the ranking explainable and
shows exactly which required skills a candidate is missing.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.feature_engineering.embeddings import get_embeddings
from src.feature_engineering.skills import extract_skills
from src.preprocessing.text_cleaning import clean_text

SEMANTIC_WEIGHT = 0.7
SKILL_WEIGHT = 0.3


def calculate_similarity(resume_embedding, jd_embedding) -> float:
    """Cosine similarity between two vectors."""
    return float(cosine_similarity([resume_embedding], [jd_embedding])[0][0])


def semantic_scores(jd_clean: str, resumes_clean: list[str]) -> np.ndarray:
    """Cosine similarity of each resume to the job description, in [0, 1]."""
    vectors = get_embeddings([jd_clean, *resumes_clean])
    sims = vectors[1:] @ vectors[0]
    return np.clip(sims, 0.0, 1.0)


def keyword_scores(jd_clean: str, resumes_clean: list[str]) -> np.ndarray:
    """TF-IDF keyword matching: the classic approach, kept as a baseline."""
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    matrix = tfidf.fit_transform([jd_clean, *resumes_clean])
    return cosine_similarity(matrix[1:], matrix[0]).ravel()


def skill_coverage(jd_skills: set[str], resume_skills: set[str]) -> float:
    if not jd_skills:
        return 0.0
    return len(jd_skills & resume_skills) / len(jd_skills)


def rank_resumes(job_description: str, resumes: dict[str, str],
                 semantic_weight: float = SEMANTIC_WEIGHT) -> pd.DataFrame:
    """Score and rank resumes for one job.

    resumes: {candidate name or file name: raw resume text}
    Returns one row per resume, best match first.
    """
    if not resumes:
        return pd.DataFrame(columns=["rank", "candidate", "final_score", "semantic_score",
                                     "skill_coverage", "matched_skills", "missing_skills"])

    jd_clean = clean_text(job_description)
    names = list(resumes)
    cleaned = [clean_text(resumes[n]) for n in names]

    semantic = semantic_scores(jd_clean, cleaned)
    jd_skills = extract_skills(jd_clean)
    rows = []
    for name, text, sem in zip(names, cleaned, semantic):
        skills = extract_skills(text)
        coverage = skill_coverage(jd_skills, skills)
        rows.append({
            "candidate": name,
            "final_score": semantic_weight * sem + (1 - semantic_weight) * coverage,
            "semantic_score": float(sem),
            "skill_coverage": coverage,
            "matched_skills": sorted(jd_skills & skills),
            "missing_skills": sorted(jd_skills - skills),
        })

    ranked = pd.DataFrame(rows).sort_values("final_score", ascending=False).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    for col in ("final_score", "semantic_score", "skill_coverage"):
        ranked[col] = ranked[col].round(3)
    return ranked
