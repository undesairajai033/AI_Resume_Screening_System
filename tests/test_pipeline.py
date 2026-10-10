"""Tests for the screening pipeline.  Run: pytest -q

A lightweight stand-in encoder replaces the real sentence-transformer, so the
tests run offline in seconds and check the pipeline logic, not the model.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import HashingVectorizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.feature_engineering import embeddings  # noqa: E402
from src.feature_engineering.skills import extract_skills  # noqa: E402
from src.ingestion.pdf_parser import extract_text  # noqa: E402
from src.models.ranking_model import rank_resumes  # noqa: E402
from src.preprocessing.text_cleaning import chunk_words, clean_text  # noqa: E402

SAMPLES = os.path.join(ROOT, "data", "samples")


class BagOfWordsEncoder:
    def __init__(self):
        self.vec = HashingVectorizer(n_features=2048, stop_words="english", alternate_sign=False)

    def encode(self, texts):
        return self.vec.transform(texts).toarray()


@pytest.fixture(autouse=True)
def fake_encoder():
    embeddings.set_encoder(BagOfWordsEncoder())
    yield
    embeddings.set_encoder(None)


def load_samples():
    jd = open(os.path.join(SAMPLES, "job_description.txt"), encoding="utf-8").read()
    folder = os.path.join(SAMPLES, "resumes")
    return jd, {f: extract_text(os.path.join(folder, f)) for f in sorted(os.listdir(folder))}


# ── cleaning & skills ────────────────────────────────────────────────────────
def test_clean_text_keeps_tech_terms():
    text = clean_text("Skills: C++, C#, .NET, Node.js, CI/CD. Email me@x.com, see https://x.io")
    for term in ["c++", "c#", ".net", "node.js", "ci/cd"]:
        assert term in text
    assert "@" not in text and "http" not in text


def test_skills_found_and_false_positives_avoided():
    skills = extract_skills(clean_text("Python, R, SQL, scikit-learn, Node.js, Power BI. I go to the office."))
    assert {"python", "r", "sql", "scikit-learn", "node.js", "power bi"} <= skills
    assert "javascript" not in skills          # "js" inside node.js is not JavaScript
    assert "go" not in skills                  # "go to the office" is not Golang


def test_long_text_is_chunked_not_truncated():
    chunks = chunk_words(" ".join(f"w{i}" for i in range(500)))
    assert len(chunks) == 3 and chunks[-1].endswith("w499")


def test_embeddings_are_unit_length():
    vecs = embeddings.get_embeddings(["python machine learning " * 150, "sales and marketing"])
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0)


# ── file reading ─────────────────────────────────────────────────────────────
def test_pdf_and_txt_extraction():
    folder = os.path.join(SAMPLES, "resumes")
    pdf = extract_text(os.path.join(folder, "ananya_rao.pdf"))
    txt = extract_text(os.path.join(folder, "rahul_mehta.txt"))
    assert "FastAPI" in pdf and "Power BI" in txt
    with open(os.path.join(folder, "ananya_rao.pdf"), "rb") as f:   # upload-style file object
        assert "FastAPI" in extract_text(f, filename="upload.pdf")


# ── ranking ──────────────────────────────────────────────────────────────────
def test_ranking_puts_ml_candidate_first_and_mechanical_last():
    jd, resumes = load_samples()
    ranked = rank_resumes(jd, resumes)
    assert ranked.iloc[0]["candidate"] == "ananya_rao.pdf"
    assert ranked.iloc[-1]["candidate"] == "vikram_nair.txt"
    assert list(ranked["rank"]) == list(range(1, len(resumes) + 1))
    assert ranked["final_score"].is_monotonic_decreasing
    top = ranked.iloc[0]
    assert {"python", "fastapi", "docker"} <= set(top["matched_skills"])


def test_empty_input():
    assert rank_resumes("python developer", {}).empty


# ── API ──────────────────────────────────────────────────────────────────────
def test_api_rank_endpoint():
    from fastapi.testclient import TestClient
    from src.api.main import app

    client = TestClient(app)
    assert client.get("/").json()["status"] == "ok"

    jd, _ = load_samples()
    folder = os.path.join(SAMPLES, "resumes")
    files = [("resumes", (f, open(os.path.join(folder, f), "rb"))) for f in sorted(os.listdir(folder))]
    res = client.post("/rank", data={"job_description": jd}, files=files)
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["count"] == 5 and body["results"][0]["candidate"] == "ananya_rao.pdf"

    bad = client.post("/rank", data={"job_description": jd},
                      files=[("resumes", ("photo.png", b"\x89PNG", "image/png"))])
    assert bad.status_code == 400


# ── evaluation maths ─────────────────────────────────────────────────────────
def test_metrics():
    from evaluate import average_precision, ndcg_at_k, precision_at_k
    perfect = np.array([1, 1, 0, 0], dtype=float)
    assert precision_at_k(perfect, 2) == 1.0
    assert ndcg_at_k(perfect, 10) == pytest.approx(1.0)
    assert average_precision(np.array([0, 1, 0, 1], dtype=float)) == pytest.approx((1 / 2 + 2 / 4) / 2)


def test_evaluate_runs_on_small_dataset():
    from evaluate import evaluate
    from evaluation.job_descriptions import JOB_DESCRIPTIONS

    rows = []
    for cat in ["Data Science", "Sales", "Civil Engineer"]:
        for i in range(4):
            rows.append({"Category": cat, "clean": clean_text(JOB_DESCRIPTIONS[cat] + f" candidate {i}")})
    results = evaluate(pd.DataFrame(rows), use_embeddings=True)
    assert set(results) == {"Keyword (TF-IDF)", "Skills only", "Semantic", "Combined"}
    assert results["Keyword (TF-IDF)"]["P@5"] > 0.5
