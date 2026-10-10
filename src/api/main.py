"""REST API for resume ranking.

Run:   uvicorn src.api.main:app --reload
Docs:  http://127.0.0.1:8000/docs  (try the /rank endpoint from the browser)
"""

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from src.ingestion.pdf_parser import extract_text
from src.models.ranking_model import rank_resumes

app = FastAPI(
    title="Resume Screening API",
    description="Rank resumes against a job description using sentence embeddings and skill matching.",
    version="1.0.0",
)

MAX_FILES = 50
MAX_FILE_BYTES = 5 * 1024 * 1024
ALLOWED = (".pdf", ".docx", ".txt")


@app.get("/")
def health():
    return {"status": "ok", "message": "Resume Screening API. See /docs."}


@app.post("/rank")
async def rank(
    job_description: str = Form(..., description="Full job description text"),
    resumes: list[UploadFile] = File(..., description="Resume files (PDF, DOCX or TXT)"),
):
    if not job_description.strip():
        raise HTTPException(400, "job_description is empty")
    if len(resumes) > MAX_FILES:
        raise HTTPException(400, f"Upload at most {MAX_FILES} resumes at a time")

    texts = {}
    for upload in resumes:
        name = upload.filename or "resume"
        if not name.lower().endswith(ALLOWED):
            raise HTTPException(400, f"{name}: only PDF, DOCX and TXT files are supported")
        data = await upload.read()
        if len(data) > MAX_FILE_BYTES:
            raise HTTPException(400, f"{name}: file is larger than 5 MB")
        try:
            texts[name] = extract_text(data, filename=name)
        except Exception:
            raise HTTPException(400, f"{name}: could not read this file")

    ranked = rank_resumes(job_description, texts)
    return {"count": len(ranked), "results": ranked.to_dict(orient="records")}
