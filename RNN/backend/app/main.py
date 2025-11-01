"""
FastAPI application for serving the RNN text generation model
Provides REST API endpoints for text generation and model information
"""

from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import os
import sys
from pathlib import Path
import traceback
import json

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
# App + CORS
# -------------------------------------------------------------------
app = FastAPI(
    title="RNN Text Generator API",
    description="REST API for generating text using LSTM-based RNN model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

FRONTEND = os.getenv("FRONTEND_ORIGIN", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND] if FRONTEND != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# All public routes live under /api so the Vercel rewrite works
api = APIRouter(prefix="/api")

# -------------------------------------------------------------------
# Model setup
# -------------------------------------------------------------------
generator: TextGenerator | None = None

# Your repo layout (per screenshot):
# RNN/backend/app/main.py
# RNN/backend/app/saved_models/{model.pt|model.h5, config.json, tokenizer.json}
HERE = Path(__file__).parent
MODEL_DIR = (HERE / "saved_models").resolve()

@app.on_event("startup")
async def load_model():
    """Load model on startup with hard-set path and loud diagnostics."""
    global generator
    try:
        print("[BOOT] Expected model directory:", MODEL_DIR)
        model_pt = MODEL_DIR / "model.pt"
        model_h5 = MODEL_DIR / "model.h5"
        cfg_path = MODEL_DIR / "config.json"
        tok_json = MODEL_DIR / "tokenizer.json"
        tok_pkl  = MODEL_DIR / "tokenizer.pkl"

        # Show what exists
        print("[BOOT] Exists?",
              "model.pt:", model_pt.exists(),
              "| model.h5:", model_h5.exists(),
              "| config.json:", cfg_path.exists(),
              "| tokenizer.json:", tok_json.exists(),
              "| tokenizer.pkl:", tok_pkl.exists())

        if not MODEL_DIR.exists():
            raise FileNotFoundError(f"saved_models dir does not exist: {MODEL_DIR}")

        if not cfg_path.exists():
            raise FileNotFoundError(f"config.json missing in {MODEL_DIR}")

        if not (tok_json.exists() or tok_pkl.exists()):
            raise FileNotFoundError(f"tokenizer.json|tokenizer.pkl missing in {MODEL_DIR}")

        # Load config
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        # Instantiate generator with config
        generator = TextGenerator(
            sequence_length=cfg.get("sequence_length", 30),
            embedding_dim=cfg.get("embedding_dim", 50),
            lstm_units=cfg.get("lstm_units", 75),
            num_lstm_layers=cfg.get("num_lstm_layers", 1),
            dropout_rate=cfg.get("dropout_rate", 0.2),
            vocab_size=cfg.get("vocab_size"),
        )
        generator.config = cfg

        # Delegate to class loader (it supports tokenizer.json or .pkl in your code)
        print(f"[BOOT] Loading model from {MODEL_DIR} …")
        generator.load_model(str(MODEL_DIR))

        # Confirm we actually have a model object
        loaded_ok = (
            getattr(generator, "model", None) is not None or
            getattr(generator, "torch_model", None) is not None
        )
        if not loaded_ok:
            raise RuntimeError("Generator loaded but no .model/.torch_model set; ensure TextGenerator.load_model sets self.model")

        print("✓ STARTUP SUCCESS — model loaded and ready.")

    except Exception as e:
        print("✗ STARTUP FAILURE — model NOT loaded.")
        print("  Reason:", e)
        print(traceback.format_exc())

# -------------------------------------------------------------------
# API ROUTES (all under /api/*)
# -------------------------------------------------------------------
@api.get("/health", response_model=HealthResponse, tags=["Health"])
async def api_health():
    is_loaded = generator is not None and (
        getattr(generator, "model", None) is not None or
        getattr(generator, "torch_model", None) is not None
    )
    return HealthResponse(
        status="healthy" if is_loaded else "model_not_loaded",
        model_loaded=is_loaded,
    )

@api.get("/diag", tags=["Health"])
async def api_diag():
    """Simple diagnostics to see what the server sees on Railway."""
    def exists(p: Path): return p.exists() and p.is_file()
    model_pt = MODEL_DIR / "model.pt"
    model_h5 = MODEL_DIR / "model.h5"
    cfg_path = MODEL_DIR / "config.json"
    tok_json = MODEL_DIR / "tokenizer.json"
    tok_pkl  = MODEL_DIR / "tokenizer.pkl"

    return {
        "cwd": str(Path.cwd()),
        "here": str(HERE),
        "model_dir": str(MODEL_DIR),
        "files": {
            "model.pt": exists(model_pt),
            "model.h5": exists(model_h5),
            "config.json": exists(cfg_path),
            "tokenizer.json": exists(tok_json),
            "tokenizer.pkl": exists(tok_pkl),
        },
        "generator_present": generator is not None,
        "has_model_attr": getattr(generator, "model", None) is not None if generator else False,
        "has_torch_model_attr": getattr(generator, "torch_model", None) is not None if generator else False,
    }

@api.get("/model-info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    if generator is None or (generator.model is None and getattr(generator, "torch_model", None) is None):
        raise HTTPException(status_code=503, detail="Model not loaded")
    if generator.config is None:
        raise HTTPException(status_code=500, detail="Model config not available")

    cfg = generator.config
    return ModelInfo(
        vocabulary_size=cfg["vocab_size"],
        sequence_length=cfg["sequence_length"],
        embedding_dim=cfg["embedding_dim"],
        lstm_units=cfg["lstm_units"],
        num_layers=cfg["num_lstm_layers"],
        is_loaded=True,
    )

@api.post("/generate", response_model=GenerateResponse, tags=["Generation"])
async def generate_text(request: GenerateRequest):
    if generator is None or (generator.model is None and getattr(generator, "torch_model", None) is None):
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        generated = generator.generate_text(
            seed_text=request.seed_text,
            num_words=request.num_words,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            use_beam_search=request.use_beam_search,
            beam_width=request.beam_width,
        )
        num_generated = max(0, len(generated.split()) - len(request.seed_text.split()))
        return GenerateResponse(
            seed_text=request.seed_text,
            generated_text=generated,
            num_words_generated=num_generated,
            temperature=request.temperature,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating text: {e}")

@api.get("/stats", tags=["Stats"])
async def get_stats():
    if generator is None or (generator.model is None and getattr(generator, "torch_model", None) is None):
        raise HTTPException(status_code=503, detail="Model not loaded")
    if generator.config is None:
        raise HTTPException(status_code=500, detail="Model config not available")

    stats = {
        "model_config": generator.config,
        "model_parameters": generator.count_params(),
        "training_history_available": generator.history is not None,
    }
    if generator.history is not None:
        h = generator.history.history
        stats["training_stats"] = {
            "final_loss": float(h["loss"][-1]),
            "final_val_loss": float(h["val_loss"][-1]),
            "epochs_trained": len(h["loss"]),
        }
    return stats

@api.get("/visualizations/architecture", tags=["Visualizations"])
async def get_architecture():
    path = "visualizations/model_architecture.png"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Architecture diagram not found.")
    return FileResponse(path=path, media_type="image/png")

@api.get("/visualizations/training", tags=["Visualizations"])
async def get_training_history():
    path = "visualizations/training_history.png"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Training history plot not found.")
    return FileResponse(path=path, media_type="image/png")

app.include_router(api)

# -------------------------------------------------------------------
# Root + error handler
# -------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "RNN Text Generator API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "api_prefix": "/api",
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "detail": str(exc)},
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
