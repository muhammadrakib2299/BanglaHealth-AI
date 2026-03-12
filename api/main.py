"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.predict import router as predict_router
from api.routes.explain import router as explain_router

app = FastAPI(
    title="BanglaHealth-AI API",
    description=(
        "Explainable AI for Patient Risk Stratification in Low-Resource Clinical Settings. "
        "Predicts patient risk levels (Low/Medium/High) for diabetes and heart disease "
        "with SHAP-based explanations."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow Streamlit and local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(predict_router)
app.include_router(explain_router)


@app.get("/", tags=["Health"])
def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "project": "BanglaHealth-AI",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/models", tags=["Health"])
def list_models():
    """List available models and their status."""
    from pathlib import Path

    models_dir = Path(__file__).resolve().parent.parent / "models"
    available = []
    for f in models_dir.glob("*.joblib"):
        if "_scaler" not in f.stem:
            parts = f.stem.split("_", 1)
            available.append({
                "dataset": parts[0],
                "model": parts[1] if len(parts) > 1 else "unknown",
                "file": f.name,
            })
    return {"models": available}
