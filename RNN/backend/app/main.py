"""
FastAPI application for serving the RNN text generation model
Provides REST API endpoints for text generation and model information
"""

from fastapi import FastAPI, HTTPException, APIRouter, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import os
import sys
from pathlib import Path
import json
import traceback

# Ensure local imports work
sys.path.insert(0, str(Path(__file__).parent))

from text_generator import TextGenerator
from models import (
    GenerateRequest,
    GenerateResponse,
    ModelInfo,
    HealthResponse,
    ErrorResponse,
)

# -------------------------------------------------------------------
# App + CORS  (works for Vercel and Railway)
# -------------------------------------------------------------------
app = FastAPI(
    title="RNN Text Generator API",
    description="REST API for generating text using LSTM-based RNN model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

ALLOWED_ORIGINS = [
    "https://cst-435-react.vercel.app",
    # include any preview domains you use:
    "https://cst-435-react-git-main-tatums-projects-965c11b1.vercel.app",
    "https://cst-435-react-n8pzza1hs-tatums-projects-965c11b1.vercel.app",
]
ORIGIN_REGEX = r"^https://cst-435-react(?:-[a-z0-9]+)?-tatums-projects-965c11b1\.vercel\.app$"

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# All public routes live under /api so your frontend should hit /api/*
api = APIRouter(prefix="/api")

# -------------------------------------------------------------------
# Model setup (no moving files needed)
# -------------------------------------------------------------------
generator: TextGenerator | None = None

HERE = Path(__file__).parent
# If this file is RNN/backend/app/main.py, then backend root is HERE.parent
BACKEND_ROOT = HERE.parent
MODEL_DIRS = [
    BACKEND_ROOT / "saved_models",  # RNN/backend/saved_models  <-- your current structure
    HERE / "saved_models",          # RNN/backend/app/saved_models
    BACKEND_ROOT.parent / "saved_models",  # RNN/saved_models (fallback)
]


@app.on_event("startup")
async def load_model():
    """Load model (Torch .pt preferred, Keras .h5 as legacy fallback)."""
    global generator
    try:
        found_dir = None
        for d in MODEL_DIRS:
            if (d / "config.json").exists() and (
                (d / "model.pt").exists() or (d / "model.h5").exists()
            ):
                found_dir = d
                break

        print("[BOOT] Candidate model dirs:", [str(p) for p in MODEL_DIRS])
        print("[BOOT] Using model dir:", found_dir if found_dir else "(not found)")

        if not found_dir:
            print("⚠ Model not found. Train/save to any saved_models path above.")
            return

        # Load config first (required)
        cfg_path = found_dir / "config.json"
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        # Build generator with config
        generator = TextGenerator(
            sequence_length=cfg.get("sequence_length", 30),
            embedding_dim=cfg.get("embedding_dim", 50),
            lstm_units=cfg.get("lstm_units", 75),
            num_lstm_layers=cfg.get("num_lstm_layers", 1),
            dropout_rate=cfg.get("dropout_rate", 0.2),
            vocab_size=cfg.get("vocab_size"),
        )
        generator.config = cfg

        # Let TextGenerator handle .pt/.h5 + tokenizer.json/.pkl
        generator.load_model(str(found_dir))
        ok = (getattr(generator, "model", None) is not None) or (
            getattr(generator, "torch_model", None) is not None
        )
        if not ok:
            raise RuntimeError("Model object missing after load")

        print("✅ MODEL LOADED")

    except Exception:
        print("❌ MODEL LOAD FAILED")
        print(traceback.format_exc())


# -------------------------------------------------------------------
# API ROUTES (all under /api/*)
# -------------------------------------------------------------------
@api.get("/health", response_model=HealthResponse)
async def health():
    ready = generator is not None and (
        getattr(generator, "model", None) is not None
        or getattr(generator, "torch_model", None) is not None
    )
    return HealthResponse(
        status="healthy" if ready else "model_not_loaded",
        model_loaded=ready,
    )


@api.get("/model-info", response_model=ModelInfo)
async def model_info():
    if generator is None or generator.config is None:
        raise HTTPException(503, "Model not loaded")
    cfg = generator.config
    return ModelInfo(
        vocabulary_size=cfg["vocab_size"],
        sequence_length=cfg["sequence_length"],
        embedding_dim=cfg["embedding_dim"],
        lstm_units=cfg["lstm_units"],
        num_layers=cfg["num_lstm_layers"],
        is_loaded=True,
    )


@api.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    if generator is None or getattr(generator, "model", None) is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    text = generator.generate_text(
        seed_text=request.seed_text,
        num_words=request.num_words,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
        use_beam_search=request.use_beam_search,
        beam_width=request.beam_width,
    )
    return GenerateResponse(
        seed_text=request.seed_text,
        generated_text=text,
        num_words_generated=max(0, len(text.split()) - len(request.seed_text.split())),
        temperature=request.temperature,
    )


@api.get("/visualizations/architecture")
async def viz_architecture():
    path = BACKEND_ROOT / "visualizations" / "model_architecture.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Architecture diagram not found.")
    return FileResponse(str(path), media_type="image/png")


@api.get("/visualizations/training")
async def viz_training():
    path = BACKEND_ROOT / "visualizations" / "training_history.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Training history plot not found.")
    return FileResponse(str(path), media_type="image/png")


app.include_router(api)

# Root + error handler
@app.get("/", include_in_schema=False)
async def root():
    return {"message": "RNN API Ready", "docs": "/docs", "api_prefix": "/api"}

@app.exception_handler(Exception)
async def handler(request: Request, exc):
    return JSONResponse(status_code=500, content={"error": str(exc)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
