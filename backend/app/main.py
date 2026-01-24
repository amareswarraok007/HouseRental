from __future__ import annotations

from pathlib import Path
from typing import Dict

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from .agents import run_incident_pipeline
from .ingest import load_csvs
from .learning_store import LearningStore
from .llm import get_llm
from .rag import IncidentRAG


BASE_DIR = Path(__file__).resolve().parents[1]
SAMPLE_DATA = BASE_DIR / "sample_data"
MEMORY_DIR = BASE_DIR / "memory"
RAG_DOCS = BASE_DIR / "rag_docs"

app = FastAPI(title="Incident AI Engine")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/run")
async def run_demo(
    metrics: UploadFile | None = File(default=None),
    logs: UploadFile | None = File(default=None),
    orders: UploadFile | None = File(default=None),
    deployments: UploadFile | None = File(default=None),
):
    if all(file is None for file in [metrics, logs, orders, deployments]):
        data = load_csvs(SAMPLE_DATA)
    else:
        data = {
            "metrics": pd.read_csv(metrics.file),
            "logs": pd.read_csv(logs.file),
            "orders": pd.read_csv(orders.file),
            "deployments": pd.read_csv(deployments.file),
        }

    rag = IncidentRAG(RAG_DOCS, MEMORY_DIR / "chroma")
    store = LearningStore(MEMORY_DIR / "learning.json")
    llm = get_llm()
    response = run_incident_pipeline(data, rag, store, llm)
    return response.model_dump()


@app.get("/api/demo")
def demo() -> Dict:
    data = load_csvs(SAMPLE_DATA)
    rag = IncidentRAG(RAG_DOCS, MEMORY_DIR / "chroma")
    store = LearningStore(MEMORY_DIR / "learning.json")
    llm = get_llm()
    response = run_incident_pipeline(data, rag, store, llm)
    return response.model_dump()
