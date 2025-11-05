from __future__ import annotations
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from starlette.status import HTTP_503_SERVICE_UNAVAILABLE

# Your existing modules:
from projects.p5.text_generator import TextGenerator
from projects.p5.models import GenerateRequest, GenerateResponse, ModelInfo

router = APIRouter(prefix="/api/p5", tags=["project-5"])

# ------------ config ------------
MODEL_DIR = os.getenv("MODEL_DIR", "projects/p5/saved_models")
MODEL_DIR_PATH = Path(MODEL_DIR).resolve()

# Single shared generator (lazy)
_gen: Optional[TextGenerator] = None
_model_loaded: bool = False
_last_error: Optional[str] = None


def _is_lfs_pointer(p: Path) -> bool:
    try:
        head = p.read_bytes()[:64]
        return head.startswith(b"version https://git-lfs.github.com/spec/v1")
    except Exception:
        return False


def _load_model_if_needed() -> None:
    global _gen, _model_loaded, _last_error
    if _model_loaded:
        return

    try:
        # Sanity on files
        weights = MODEL_DIR_PATH / "model.pt"
        tok = MODEL_DIR_PATH / "tokenizer.json"
        cfg = MODEL_DIR_PATH / "config.json"

        missing = [str(p.name) for p in (weights, tok, cfg) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing model artifacts in {MODEL_DIR_PATH}: {', '.join(missing)}"
            )

        if _is_lfs_pointer(weights):
            raise RuntimeError(
                f"{weights} appears to be a Git LFS pointer, not real weights. "
                "Ensure Railway build pulls LFS blobs: `git lfs install && git lfs pull`."
            )

        # Create and load
        _gen = TextGenerator()
        _gen.load_model(str(MODEL_DIR_PATH))
        _model_loaded = True
        _last_error = None
        print(f"[p5] Model loaded from {MODEL_DIR_PATH}")
    except Exception as e:
        _model_loaded = False
        _last_error = f"{type(e).__name__}: {e}"
        # Keep stderr log detailed; API returns safe message
        print(f"[p5] Failed to load model: {_last_error}")


# ----------- routes ------------

@router.get("/health")
def health():
    # Try loading once if not loaded
    if not _model_loaded:
        _load_model_if_needed()
    return {
        "status": "ok" if _model_loaded else "not-ready",
        "model_loaded": _model_loaded,
        "model_dir": str(MODEL_DIR_PATH),
        "last_error": _last_error,
    }


@router.post("/warmup")
def warmup():
    """Force load the model and return status."""
    _load_model_if_needed()
    if not _model_loaded:
        raise HTTPException(status_code=500, detail=_last_error or "Unknown load error")
    return {"ok": True, "model_loaded": True, "model_dir": str(MODEL_DIR_PATH)}


@router.get("/model-info", response_model=ModelInfo)
def model_info():
    if not _model_loaded:
        _load_model_if_needed()
    if not _model_loaded or _gen is None or _gen.model is None:
        # Return a helpful error but as 503 (unavailable)
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail=_last_error or "Model not loaded yet."
        )

    # Build a ModelInfo from the generator’s config
    vocab_size = (
        (len(getattr(_gen.tokenizer, "word_index", {})) + 1)
        if getattr(_gen, "tokenizer", None) and getattr(_gen.tokenizer, "word_index", None)
        else (_gen.vocab_size or 0)
    )
    return ModelInfo(
        vocabulary_size=int(vocab_size),
        sequence_length=int(getattr(_gen, "sequence_length", 0) or 0),
        embedding_dim=int(getattr(_gen, "embedding_dim", 0) or 0),
        lstm_units=int(getattr(_gen, "lstm_units", 0) or 0),
        num_layers=int(getattr(_gen, "num_lstm_layers", 0) or 0),
        is_loaded=True,
    )


@router.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if not _model_loaded:
        _load_model_if_needed()
    if not _model_loaded or _gen is None or _gen.model is None:
        # 503 here is what you saw; include actionable detail
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail=_last_error or "Model not loaded. Check /api/p5/health and MODEL_DIR."
        )

    try:
        text = _gen.generate_text(
            seed_text=req.seed_text,
            num_words=req.num_words,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            use_beam_search=req.use_beam_search,
            beam_width=req.beam_width,
        )
        return GenerateResponse(
            seed_text=req.seed_text,
            generated_text=text,
            num_words_generated=req.num_words,
            temperature=req.temperature,
        )
    except Exception as e:
        # Surface a 500 with concise error
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")
