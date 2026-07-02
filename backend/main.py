"""FastAPI application: exposes /api/grade and serves the static frontend."""
from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.grading import grade_submission
from backend.models import GradeRequest, GradeReport

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("grader")

app = FastAPI(title="AI Assignment Grader")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/grade", response_model=GradeReport)
def grade(request: GradeRequest) -> GradeReport:
    """Grade a single submission against the book + rubric."""
    logger.info("Grading request received for student=%s (%d chars)",
                request.student_name, len(request.answer_text))
    try:
        report = grade_submission(request.student_name, request.answer_text)
    except RuntimeError as exc:
        logger.error("Grading failed: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    logger.info("Graded %s: %s/%s (flags=%d)", request.student_name,
                report.total_score, report.max_score, len(report.flags))
    return report


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "index_ready": settings.vector_store_dir.exists()}


# Serve the single-page frontend at "/"
app.mount("/", StaticFiles(directory=str(settings.book_path.parent.parent / "frontend"), html=True), name="frontend")
