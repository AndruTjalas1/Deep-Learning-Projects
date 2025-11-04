# backend/routers/p5.py
from fastapi import APIRouter, HTTPException
from pathlib import Path
from typing import Optional, Dict, Any
import os, json, traceback, importlib

# Pull your request/response models from projects/p5/models.py
P5_MODELS = importlib.import_module("projects.p5.models")
GenerateRequest = getattr(P5_MODELS, "GenerateRequest")
GenerateResponse = getattr(P5_MODELS, "GenerateResponse")
ModelInfo = getattr(P5_MODELS, "ModelInfo")
HealthResponse = getattr(P5_MODELS, "HealthResponse")

router = APIRouter(prefix="/api/p5", tags=["p5 (RNN)"])

# ------------------ Model resolution (from your old main.py) ------------------
def _candidate_model_dirs():
    paths = []
    env_dir = os.getenv("MODEL_DIR") or os.getenv("RNN_MODEL_DIR")
    if env_dir:
        try:
            paths.append(Path(env_dir).expanduser().resolve())
        except Exception:
            pass
    here = Path(__file__).resolve()
    paths += [
        here.parents[2] / "projects" / "p5" / "saved_models",  # backend/projects/p5/saved_models
        Path.cwd() / "saved_models",                           # fallback
    ]
    # de-dupe
    out, seen = [], set()
    for p in map(lambda x: x.resolve(), paths):
        if str(p) not in seen:
            out.append(p)
            seen.add(str(p))
    return out

def _pick_model_dir() -> Optional[Path]:
    searched = []
    for d in _candidate_model_dirs():
        searched.append(str(d))
        if d.is_dir() and (d / "config.json").exists() and (d / "tokenizer.json").exists():
            return d
    print("[P5] Model dir not found. Searched:", searched, flush=True)
    return None

def _is_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            head = f.read(64)
        return head.startswith(b"version https://git-lfs.github.com/spec/v1")
    except Exception:
        return False

# ------------------ Lazy Singleton ------------------
_generator = None
_cfg: Optional[Dict[str, Any]] = None
_MODEL_DIR: Optional[Path] = _pick_model_dir()

def _ensure_loaded():
    global _generator, _cfg
    if _generator is not None:
        return

    if _MODEL_DIR is None:
        raise HTTPException(503, "No model directory found (config.json + tokenizer.json missing)")

    from projects.p5.text_generator import TextGenerator  # your implementation

    cfg_path = _MODEL_DIR / "config.json"
    model_pt = _MODEL_DIR / "model.pt"

    if not cfg_path.exists():
        raise HTTPException(503, f"Missing config.json at {_MODEL_DIR}")
    if not model_pt.exists():
        raise HTTPException(503, f"Missing model.pt at {_MODEL_DIR}")
    if _is_lfs_pointer(model_pt):
        raise HTTPException(500, "model.pt is a Git LFS pointer; real weights not present.")

    _cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    # Instantiate using your config keys (matches your text_generator defaults)
    _generator = TextGenerator(
        sequence_length=_cfg.get("sequence_length", 50),
        embedding_dim=_cfg.get("embedding_dim", 100),
        lstm_units=_cfg.get("lstm_units", 150),
        num_lstm_layers=_cfg.get("num_lstm_layers", 2),
        dropout_rate=_cfg.get("dropout_rate", 0.2),
        vocab_size=_cfg.get("vocab_size"),
    )
    _generator.config = _cfg  # keep parity with your training code
    _generator.load_model(str(_MODEL_DIR))
    # If you later add optimize_for_inference() in your class, you can call it here.

# ------------------ Routes ------------------
@router.get("/health", response_model=HealthResponse)
def health():
    try:
        _ensure_loaded()
        return HealthResponse(status="healthy", model_loaded=True)
    except HTTPException as e:
        return HealthResponse(status="error", model_loaded=False)
    except Exception:
        print(traceback.format_exc(), flush=True)
        return HealthResponse(status="error", model_loaded=False)

@router.get("/model-info", response_model=ModelInfo)
def model_info():
    _ensure_loaded()
    cfg = _cfg or {}
    return ModelInfo(
        vocabulary_size=cfg.get("vocab_size") or 0,
        sequence_length=cfg.get("sequence_length") or 0,
        embedding_dim=cfg.get("embedding_dim") or 0,
        lstm_units=cfg.get("lstm_units") or 0,
        num_layers=cfg.get("num_lstm_layers") or 0,
        is_loaded=True,
    )

@router.post("/generate", response_model=GenerateResponse)
def generate_text(request: GenerateRequest):
    _ensure_loaded()
    try:
        text = _generator.generate_text(
            seed_text=request.seed_text,
            num_words=request.num_words,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            use_beam_search=request.use_beam_search,
            beam_width=request.beam_width,
        )
        # Your GenerateResponse expects these fields:
        return GenerateResponse(
            seed_text=request.seed_text,
            generated_text=text,
            num_words_generated=max(0, len(text.split()) - len(request.seed_text.split())),
            temperature=request.temperature,
        )
    except Exception as e:
        raise HTTPException(500, f"Generation failed: {e}")
