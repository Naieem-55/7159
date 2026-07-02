FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Build the vector index at container start (requires OPENAI_API_KEY at runtime),
# then launch the API server.
CMD ["sh", "-c", "python -m scripts.build_index && uvicorn backend.main:app --host 0.0.0.0 --port 8000"]
