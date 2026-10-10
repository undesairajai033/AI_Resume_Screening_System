# 📄 AI Resume Screening System

Ranks a stack of resumes against a job description by **meaning**, not just matching keywords, and shows which required skills each candidate has and which they're missing. A resume that says "built REST services in FastAPI" still matches a job asking for "Python backend API development".

**Python · Sentence Transformers · scikit-learn · FastAPI · Streamlit · pdfplumber**

<!-- Screenshot: add screenshots/dashboard.png and uncomment
![Dashboard](screenshots/dashboard.png) -->

---

## ✨ What it does

- Upload any number of resumes (**PDF, DOCX or TXT**) and paste a job description
- Get a **ranked shortlist** with a score for each candidate
- See **matched and missing skills** per candidate, so every ranking is explainable
- Adjust how much weight goes to meaning versus required skills
- Download the results as CSV
- Use the same ranking from code through a **REST API**

## 🧠 How it works

```
Resumes ─► extract text ─► clean ─► chunk ─► sentence embeddings ─┐
                                 └──────► skill extraction ───────┤
Job description ─► clean ─► embeddings + skill extraction ────────┴─► score ─► ranked shortlist

score = 0.7 × semantic similarity + 0.3 × share of the job's skills found in the resume
```

- **Semantic similarity:** `all-MiniLM-L6-v2` sentence embeddings with cosine similarity. The model reads about 256 tokens at a time, so long resumes are split into overlapping chunks and averaged. Otherwise everything after the first page would be silently ignored.
- **Skill coverage:** a vocabulary of ~100 technical and business skills with common aliases (e.g. `sklearn` → scikit-learn, `k8s` → Kubernetes). It also guards against false matches, so "Node.js" doesn't count as JavaScript and "I go to…" doesn't count as Golang.

## 📊 Evaluation

`python evaluate.py` measures how well each method finds the right candidates. It uses the public Kaggle **Resume Dataset**, where each resume is labelled with one of 25 job categories (Data Science, Java Developer, HR, Civil Engineer, …).

- For each category, a realistic job description ranks **all** resumes; resumes from that category are the correct answers.
- Duplicate resumes in the raw file are removed first.
- Metrics: Precision@5, Precision@10, NDCG@10 and MAP, averaged over the 25 jobs.

| Method | P@5 | P@10 | NDCG@10 | MAP |
|---|---|---|---|---|
| Keyword matching (TF-IDF baseline) | _run evaluate.py_ | | | |
| Skills only | | | | |
| Semantic (embeddings) | | | | |
| **Combined (used in the app)** | | | | |

## 🚀 Run it locally

```bash
git clone https://github.com/undesairajai033/AI_Resume_Screening_System.git
cd AI_Resume_Screening_System
python -m venv venv
venv\Scripts\activate                 # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt

streamlit run dashboard/app.py        # the dashboard; toggle "Use sample data" to try it
uvicorn src.api.main:app --reload     # the API; open http://127.0.0.1:8000/docs
python evaluate.py                    # reproduce the evaluation table
pytest -q                             # run the tests
```

The first run downloads the embedding model (~90 MB).

### API example

```bash
curl -X POST http://127.0.0.1:8000/rank \
  -F "job_description=Junior ML engineer with Python, scikit-learn and FastAPI" \
  -F "resumes=@data/samples/resumes/ananya_rao.pdf" \
  -F "resumes=@data/samples/resumes/priya_shah.pdf"
```

## 📁 Project structure

```
├── dashboard/app.py                  Streamlit recruiter dashboard
├── src/
│   ├── api/main.py                   FastAPI service (POST /rank)
│   ├── ingestion/pdf_parser.py       PDF / DOCX / TXT text extraction
│   ├── preprocessing/text_cleaning.py cleaning and chunking
│   ├── feature_engineering/
│   │   ├── embeddings.py             sentence embeddings (chunked, normalised)
│   │   └── skills.py                 skill vocabulary and extraction
│   └── models/ranking_model.py       scoring and ranking
├── evaluate.py                       offline evaluation
├── evaluation/job_descriptions.py    one job description per category
├── data/samples/                     fictional resumes and a sample job description
└── tests/test_pipeline.py            tests (run offline with a stand-in encoder)
```

## ⚖️ Limitations

- Similarity scores rank candidates; they aren't hiring decisions. A human should review every shortlist.
- Skill matching uses a fixed vocabulary, so niche tools need to be added to `skills.py`.
- Scanned (image-only) PDFs have no extractable text and are skipped with a warning.

## 📌 Next steps

- An LLM-written summary of why each candidate ranks where they do
- Section-aware parsing (experience vs. education vs. skills)
- Years-of-experience extraction
