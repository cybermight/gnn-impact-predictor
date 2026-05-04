"""
Early-Stage Scientific Influence Prediction API
Deeksha Saraswat | DEI Agra | 2026
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn

from app.predictor import PaperPredictor

# ── App Setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Scientific Influence Predictor API",
    description="Predicts whether a CS paper will be high-impact using Temporal-Structural GNN",
    version="1.0.0"
)

# Allow Flutter app (and any client) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load predictor once on startup
predictor = PaperPredictor()


# ── Request / Response Schemas ─────────────────────────────────────────────────
class PaperInput(BaseModel):
    title: str = Field(..., example="Attention Is All You Need")
    abstract: str = Field(..., example="We propose a new simple network architecture...")
    publication_year: int = Field(..., ge=1990, le=2024, example=2017)
    early_citation_count: int = Field(..., ge=0, example=12,
        description="Number of citations received within ~1 year of publication")
    reference_count: int = Field(..., ge=0, example=34,
        description="Number of references this paper makes")
    venue: Optional[str] = Field(None, example="NeurIPS")

class PredictionResponse(BaseModel):
    is_high_impact: bool
    confidence: float                  # 0.0 – 1.0
    impact_label: str                  # "High Impact" | "Low Impact"
    feature_summary: dict              # key structural signals shown to the user
    explanation: str                   # human-readable reason


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Scientific Influence Predictor is running!"}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(paper: PaperInput):
    """
    Takes early-stage paper metadata and returns a high-impact prediction.
    Uses structural + temporal heuristics that mirror the Weighted GCN pipeline.
    """
    try:
        result = predictor.predict(paper)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info", tags=["Info"])
def model_info():
    """Returns information about the underlying model and dataset."""
    return {
        "model": "Weighted GCN with exponential decay edge weights",
        "dataset": "OGBN-Arxiv (OGB)",
        "label_definition": "Top 10% by 5-year citation count, ranked within publication year",
        "features_used": [
            "128-dim skip-gram content embeddings",
            "In-degree (early citation count)",
            "Out-degree (reference count)",
            "PageRank score",
            "Citation velocity (first-year rate)",
            "Citation recency score",
            "Year-normalized citation rank"
        ],
        "edge_weighting": "exp(-Δt) where Δt = year gap between citing and cited paper",
        "test_auc": 0.9664,
        "test_recall": 0.9113,
    }


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)