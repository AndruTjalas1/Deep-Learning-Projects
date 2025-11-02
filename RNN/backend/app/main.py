"""
FastAPI application for serving the RNN text generation model
Provides REST API endpoints for text generation and model information
"""

from fastapi import FastAPI, HTTPException, APIRouter, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os, sys, json, traceback
from pathlib import Path

# Make local imports work
sys.path.insert(0, str(Path(__file__).parent))

from text_generator import TextGenerator
from models import (
    GenerateRequest,
    GenerateResponse,
    ModelInfo,
    HealthResponse,
)

# --------------------------- App + CORS ---------------------------
app = FastAPI(
    title="RNN Text Generator API",
    description="REST API for generating text using LSTM-based RNN model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

ALLOWED_ORIGINS = [
    "https://cst-435-react.vercel.app",
    "http://localhost:5173",
    # any preview deploys you want to allow can go here…
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

api = APIRouter(prefix="/api")

# ----------------------- Model directory picking ------------------
HERE = Path(__file__).parent.resolve()
generator: TextGenerator | None = None

def _candidate_model_dirs():
    paths = []
    env1 = os.getenv("MODEL_DIR")
    env2 = os.getenv("RNN_MODEL_DIR")
    if env1: paths.append(Path(env1))
    if env2: paths.append(Path(env2))
    paths += [
        (HERE / "saved_models"),                 # app/saved_models
        (HERE.parent / "saved_models"),          # backend/saved_models
        (HERE.parent.parent / "saved_models"),   # RNN/saved_models
        (Path.cwd() / "saved_models"),           # working dir fallback
    ]
    # normalize + de-dup
    out, seen = [], set()
    for p in paths:
        rp = p.expanduser().resolve()
        if rp not in seen:
            out.append(rp); seen.add(rp)
    return out

def _pick_model_dir():
    print("[BOOT] ENV MODEL_DIR     =", os.getenv("MODEL_DIR"), flush=True)
    print("[BOOT] ENV RNN_MODEL_DIR =", os.getenv("RNN_MODEL_DIR"), flush=True)
    for d in _candidate_model_dirs():
        if (d / "config.json").exists() and (d / "tokenizer.json").exists():
            print("[BOOT] Using model directory:", d, flush=True)
            return d
    return None

MODEL_DIR = _pick_model_dir()

# --------------------------- Startup hook -------------------------
@app.on_event("startup")
async def load_model():
    global generator
    try:
        if MODEL_DIR is None:
            raise FileNotFoundError("No suitable saved_models directory found")

        cfg_path = MODEL_DIR / "config.json"
        tok_json = MODEL_DIR / "tokenizer.json"
        model_pt = MODEL_DIR / "model.pt"

        print(
            "[BOOT] Final MODEL_DIR   =", MODEL_DIR,
            "\n[BOOT] Exists? ",
            "model.pt:", model_pt.exists(),
            "| config.json:", cfg_path.exists(),
            "| tokenizer.json:", tok_json.exists(),
            flush=True,
        )

        if not cfg_path.exists():
            raise FileNotFoundError("config.json missing")

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        generator = TextGenerator(
            sequence_length=cfg.get("sequence_length", 30),
            embedding_dim=cfg.get("embedding_dim", 50),
            lstm_units=cfg.get("lstm_units", 75),
            num_lstm_layers=cfg.get("num_lstm_layers", 1),
            dropout_rate=cfg.get("dropout_rate", 0.2),
            vocab_size=cfg.get("vocab_size"),
        )
        generator.config = cfg

        print(f"[BOOT] Loading model from {MODEL_DIR} …", flush=True)
        generator.load_model(str(MODEL_DIR))
        print("✅ MODEL LOADED SUCCESSFULLY", flush=True)

    except Exception:
        print("❌ MODEL LOAD FAILED", flush=True)
        print(traceback.format_exc(), flush=True)

# ----------------------------- Routes -----------------------------
def _is_ready() -> bool:
    return (
        generator is not None and
        (getattr(generator, "model", None) is not None or getattr(generator, "torch_model", None) is not None)
    )

# canonical
@api.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="healthy" if _is_ready() else "model_not_loaded",
                          model_loaded=_is_ready())

@api.get("/model-info", response_model=ModelInfo)
async def model_info():
    if not _is_ready() or generator is None or generator.config is None:
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
async def generate(request: GenerateRequest):
    if not _is_ready() or generator is None:
        raise HTTPException(503, "Model not loaded")
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

# aliases for legacy callers (so /generate and /stats stop 404s)
@app.get("/health")
async def health_alias():
    return await health()

@app.get("/model/info")
async def model_info_alias():
    return await model_info()

@app.post("/generate")
async def generate_alias(request: Request):
    body = await request.json()
    try:
        req = GenerateRequest(**body)
    except Exception:
        raise HTTPException(400, "Bad payload for /generate")
    resp = await generate(req)
    return resp

@app.get("/stats")
async def stats():
    # Minimal stats to satisfy callers that probe /stats
    return {
        "model_loaded": _is_ready(),
        "model_dir": str(MODEL_DIR) if MODEL_DIR else None,
        "vocab_size": getattr(generator, "vocab_size", None) or (generator.config.get("vocab_size") if generator and generator.config else None),
    }

# root + router
@app.get("/", include_in_schema=False)
async def root():
    return {"message": "RNN API Ready", "docs": "/docs", "health": "/api/health"}

app.include_router(api)

# global error handler
@app.exception_handler(Exception)
async def handler(request: Request, exc):
    return JSONResponse(status_code=500, content={"error": str(exc)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
