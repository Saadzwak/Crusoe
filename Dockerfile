# PRAETOR demo image — FastAPI backend + CureWatch UI (LEGION / X-ray / passport)
# served as one process. Public-demo defaults: MOCK_LLM=1 (deterministic,
# free, safe to share) baked in via render.yaml — override there, not here.
FROM python:3.12-slim

WORKDIR /app

# System deps for the CPU torch wheel + build tools some transitive deps need
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Runtime-only physical-layer deps (CPU torch — same pin the repo documents in
# requirements.txt) so CP-07 shows the REAL MH-PINN reconstruction, not the
# linear stand-in. pandas is required transitively (pinn/inference.py loads
# the C-MAPSS normalization reference through it). Training/exploration-only
# deps (scipy/matplotlib/py7zr/tabulate) are intentionally skipped.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir numpy pandas

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend ./backend
COPY pinn ./pinn
COPY .env.example ./.env.example

# Runtime state lives in a container-local file — ephemeral by design (a
# restart clears any demo advisory backlog, which is the behavior we want
# for a public link: it always opens clean).
ENV AGENT_DB_PATH=/app/backend/agent/praetor_state.db
ENV PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["sh", "-c", "uvicorn backend.agent.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
