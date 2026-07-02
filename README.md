# AI Assignment Grader

A small app that grades a student's ML assignment against a course book and a rubric,
using retrieval-augmented generation so the LLM only scores what the book actually supports.

## Architecture

```
book.txt --> chunk (chapter-aware, paragraph-based) --> OpenAI embeddings --> ChromaDB
                                                                                  |
rubric.md questions --> per-question retrieval query -----------------------------
                                                                                  |
student answer --> [wrapped as inert data] --+                                   v
                                              |                        retrieved evidence
                                              +--> LLM grading call (structured JSON)
                                                            |
                                                            v
                                              LLM checker call (re-validates scores,
                                              catches injection-inflated marks)
                                                            |
                                                            v
                                                  Structured grade report (API + UI)
```

- **Backend**: FastAPI (`backend/`), serving `/api/grade` and the static frontend.
- **Vector DB**: ChromaDB, persisted locally under `vector_store/`.
- **Embeddings & grading**: OpenAI (`text-embedding-3-small`, `gpt-4o-mini` by default).
- **Frontend**: single static HTML/CSS/JS page (`frontend/index.html`) — no build step.
- **Retrieval strategy**: each rubric question is used as its own retrieval query (a
  lightweight form of agentic retrieval — five targeted searches instead of one generic
  one), and the results are merged and deduplicated before grading.
- **Grading strategy**: one structured "grader" LLM call, followed by a second
  "checker" LLM call that re-validates the scores against the rubric and evidence, and
  corrects anything that looks inflated by injected text rather than by evidence.

## Prompt-injection protection

Two independent layers:

1. **Heuristic scan** (`backend/security.py`) flags known manipulation phrases
   ("ignore previous instructions", "give me full marks", "you are now...", etc.) and
   surfaces them as a flag in the report, regardless of what the LLM does.
2. **Structural defense in the prompt**: the student's text is wrapped in explicit
   `<STUDENT_SUBMISSION_DO_NOT_EXECUTE_AS_INSTRUCTIONS>` delimiters, and the system
   prompt tells the model to treat everything inside as data to grade, never as
   commands — even if it contains phrases that look like instructions. The checker
   pass re-verifies this afterwards.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # or your preferred env tool
pip install -r requirements.txt
cp .env.example .env   # then edit .env and paste your OPENAI_API_KEY
```

## Build the vector index (run once, or whenever book.txt changes)

```bash
python -m scripts.build_index
```

## Run the app

```bash
uvicorn backend.main:app --reload
```

Then open `http://localhost:8000` — paste an answer (or click one of the sample-student
buttons) and click **Grade Assignment**.

## Batch-grade the 3 sample assignments from the command line

```bash
python -m scripts.grade_samples
```

This grades every file in `assignments/` and writes a Markdown + JSON report per student
into `reports/`, plus a `summary.json`.

## Run tests

```bash
pytest
```

Tests cover chunking logic and the prompt-injection heuristics — no API key or network
access is required to run them.

## Docker

```bash
docker build -t ai-grader .
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... ai-grader
```

## Known limitation: the indexed book excerpt is partial

`data/book.txt` currently only contains the introductory section of Chapter 4 (Linear
Regression) that was available at build time. It does **not** yet include the book's
coverage of R², Ridge/Lasso/ElasticNet, RANSAC, the sigmoid function, or the `C`
parameter — the exact concepts the 5 assignment questions ask about.

This isn't a bug — it's the RAG system behaving correctly with the data it was given:
retrieval genuinely finds no supporting evidence for those claims, so the grader
correctly declines to reward them and flags them as unsupported rather than trusting
outside knowledge. **To get meaningful, differentiated scores between Student A (mostly
correct), Student B (vague/thin), and Student C (factually wrong + injection attempt),
paste the fuller book text — the sections on R², regularization, RANSAC, logistic
regression/sigmoid, and the `C` parameter — into `data/book.txt`, then re-run
`python -m scripts.build_index`.**

## Project structure

```
backend/
  config.py      settings from environment variables
  models.py      Pydantic request/response schemas
  indexing.py    book loading, chunking, embedding, ChromaDB indexing
  retrieval.py   semantic search over the indexed book
  security.py    prompt-injection detection + safe wrapping
  grading.py     retrieval orchestration + grader/checker LLM calls
  main.py        FastAPI app
frontend/
  index.html     single-page UI (no build step)
scripts/
  build_index.py   CLI: (re)build the vector index
  grade_samples.py CLI: batch-grade assignments/*.txt into reports/
data/
  book.txt       indexed course material
  rubric.md      grading rubric + questions
assignments/
  student_a.txt, student_b.txt, student_c.txt   sample submissions
tests/
  test_chunking.py, test_security.py
```

## Where this sits against the assignment's levels

- **Level 1/2**: done — vector DB, per-criterion scoring with book references, all 3
  samples gradeable, unsupported-claim handling.
- **Level 3 (partial)**: retrieval is targeted per rubric question rather than a single
  fixed query, which is a lightweight version of "the agent decides what to look up."
  It doesn't yet do iterative re-querying based on the grader's own uncertainty.
- **Level 4 (partial)**: a second checker pass reviews/corrects the draft grade, batch
  grading exists, prompt-injection protection is implemented, and logging is in place.
  The frontend is intentionally minimal rather than a full framework build — functional
  over decorative, per the brief.
