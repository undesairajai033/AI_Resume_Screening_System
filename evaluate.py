"""Measure how well each scoring method shortlists the right resumes.

Data: the public Kaggle "Resume Dataset" (UpdatedResumeDataSet.csv), where each
resume is labelled with a job category. For every category, a realistic job
description (evaluation/job_descriptions.py) is used to rank ALL resumes; the
resumes from that category are the correct answers.

Methods compared:
  * Keyword (TF-IDF)   - classic keyword matching, the baseline
  * Skills only        - share of the job's skills found in the resume
  * Semantic           - sentence-embedding similarity (all-MiniLM-L6-v2)
  * Combined           - what the app uses: 0.7 x semantic + 0.3 x skills

Duplicate resumes (the raw file repeats many) are removed first, so a method
can't score well just by finding copies of the same document.

Run:  python evaluate.py
      (first run downloads the dataset and the ~90 MB model)
Writes results/results.json and prints a results table.
"""

import json
import os
import sys
import urllib.request

import numpy as np
import pandas as pd

from evaluation.job_descriptions import JOB_DESCRIPTIONS
from src.feature_engineering.embeddings import get_embeddings
from src.feature_engineering.skills import extract_skills
from src.models.ranking_model import SEMANTIC_WEIGHT, keyword_scores, skill_coverage
from src.preprocessing.text_cleaning import clean_text

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, "data", "eval", "UpdatedResumeDataSet.csv")
RESULTS_PATH = os.path.join(ROOT, "results", "results.json")
# Public copy of the Kaggle dataset. You can also download it from
# https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset and save it to DATA_PATH.
DATA_URL = "https://huggingface.co/datasets/Unknown92/Resume_dataset/resolve/main/UpdatedResumeDataSet.csv"


# ── metrics ──────────────────────────────────────────────────────────────────
def precision_at_k(ranked_relevant: np.ndarray, k: int) -> float:
    return float(ranked_relevant[:k].mean())


def ndcg_at_k(ranked_relevant: np.ndarray, k: int) -> float:
    gains = ranked_relevant[:k] / np.log2(np.arange(2, min(k, len(ranked_relevant)) + 2))
    ideal = np.sort(ranked_relevant)[::-1][:k] / np.log2(np.arange(2, min(k, len(ranked_relevant)) + 2))
    return float(gains.sum() / ideal.sum()) if ideal.sum() else 0.0


def average_precision(ranked_relevant: np.ndarray) -> float:
    hits = np.cumsum(ranked_relevant)
    precisions = hits / np.arange(1, len(ranked_relevant) + 1)
    return float((precisions * ranked_relevant).sum() / ranked_relevant.sum()) if ranked_relevant.sum() else 0.0


def score_ranking(scores: np.ndarray, relevant: np.ndarray) -> dict:
    order = np.argsort(-scores, kind="stable")
    r = relevant[order].astype(float)
    return {"P@5": precision_at_k(r, 5), "P@10": precision_at_k(r, 10),
            "NDCG@10": ndcg_at_k(r, 10), "MAP": average_precision(r)}


# ── data ─────────────────────────────────────────────────────────────────────
def load_dataset() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        print("Downloading resume dataset...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    df["clean"] = df["Resume"].astype(str).map(clean_text)
    df = df.drop_duplicates(subset="clean").reset_index(drop=True)
    df = df[df["Category"].isin(JOB_DESCRIPTIONS)].reset_index(drop=True)
    return df


# ── evaluation ───────────────────────────────────────────────────────────────
def evaluate(df: pd.DataFrame, use_embeddings: bool = True) -> dict:
    resumes = df["clean"].tolist()
    resume_skills = [extract_skills(t) for t in resumes]
    categories = sorted(df["Category"].unique())
    jds = {c: clean_text(JOB_DESCRIPTIONS[c]) for c in categories}

    if use_embeddings:
        print(f"Embedding {len(resumes)} resumes and {len(jds)} job descriptions...")
        resume_vecs = get_embeddings(resumes)
        jd_vecs = dict(zip(categories, get_embeddings(list(jds.values()))))

    per_method = {m: [] for m in ["Keyword (TF-IDF)", "Skills only", "Semantic", "Combined"]}
    for cat in categories:
        relevant = (df["Category"] == cat).to_numpy()
        jd_skills = extract_skills(jds[cat])
        skills = np.array([skill_coverage(jd_skills, s) for s in resume_skills])

        per_method["Keyword (TF-IDF)"].append(score_ranking(keyword_scores(jds[cat], resumes), relevant))
        per_method["Skills only"].append(score_ranking(skills, relevant))
        if use_embeddings:
            semantic = np.clip(resume_vecs @ jd_vecs[cat], 0, 1)
            combined = SEMANTIC_WEIGHT * semantic + (1 - SEMANTIC_WEIGHT) * skills
            per_method["Semantic"].append(score_ranking(semantic, relevant))
            per_method["Combined"].append(score_ranking(combined, relevant))

    return {m: {k: round(float(np.mean([r[k] for r in rows])), 4) for k in rows[0]}
            for m, rows in per_method.items() if rows}


def main():
    use_embeddings = "--keyword-only" not in sys.argv
    df = load_dataset()
    print(f"{len(df)} unique resumes across {df['Category'].nunique()} job categories\n")

    results = evaluate(df, use_embeddings)

    print("| Method | P@5 | P@10 | NDCG@10 | MAP |\n|---|---|---|---|---|")
    for name, m in results.items():
        print(f"| {name} | {m['P@5']:.3f} | {m['P@10']:.3f} | {m['NDCG@10']:.3f} | {m['MAP']:.3f} |")

    if "Combined" in results:
        base, best = results["Keyword (TF-IDF)"]["NDCG@10"], results["Combined"]["NDCG@10"]
        print(f"\nCombined vs keyword baseline, NDCG@10: {(best / base - 1) * 100:+.1f}%")

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump({"resumes": len(df), "categories": int(df["Category"].nunique()),
                   "semantic_weight": SEMANTIC_WEIGHT, "results": results}, f, indent=2)
    print(f"Saved {RESULTS_PATH}")


if __name__ == "__main__":
    main()
