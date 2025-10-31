"""
FastAPI application for serving the RNN text generation model
Provides REST API endpoints for text generation and model information
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from text_generator import TextGenerator
from models import (
    GenerateRequest,
    GenerateResponse,
    ModelInfo,
    HealthResponse,
    ErrorResponse
)

# Initialize FastAPI app
app = FastAPI(
    title="RNN Text Generator API",
    description="REST API for generating text using LSTM-based RNN model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global generator instance
generator = None

# Get correct path to saved models (handle both direct and app/ subdirectory execution)
_current_dir = Path(__file__).parent
_backend_dir = _current_dir.parent if _current_dir.name == 'app' else _current_dir
MODEL_PATH = _backend_dir / "saved_models" / "model.h5"
TOKENIZER_PATH = _backend_dir / "saved_models" / "tokenizer.pkl"
CONFIG_PATH = _backend_dir / "saved_models" / "config.json"


@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    global generator
    
    try:
        # Try multiple possible paths
        possible_model_dirs = [
            str(MODEL_PATH.parent),  # Direct saved_models path
            "saved_models",           # Relative from app/
            "../saved_models",        # Up one directory from app/
        ]
        
        model_loaded = False
        loaded_config = None
        
        for model_dir in possible_model_dirs:
            model_h5 = Path(model_dir) / "model.h5"
            model_pt = Path(model_dir) / "model.pt"
            config_file = Path(model_dir) / "config.json"

            # Prefer PyTorch saved model (model.pt) if present
            if model_pt.exists():
                # Try to load config if available
                loaded_config = None
                if config_file.exists():
                    import json
                    with open(config_file, 'r') as f:
                        loaded_config = json.load(f)

                # If no config, we will let TextGenerator.load_model handle tokenizer rebuild
                seq_len = loaded_config.get('sequence_length', 30) if loaded_config else 30
                emb_dim = loaded_config.get('embedding_dim', 50) if loaded_config else 50
                lstm_u = loaded_config.get('lstm_units', 75) if loaded_config else 75
                num_layers = loaded_config.get('num_lstm_layers', 1) if loaded_config else 1
                dropout = loaded_config.get('dropout_rate', 0.2) if loaded_config else 0.2
                vocab_size = loaded_config.get('vocab_size') if loaded_config else None

                # Create generator with available parameters
                generator = TextGenerator(
                    sequence_length=seq_len,
                    embedding_dim=emb_dim,
                    lstm_units=lstm_u,
                    num_lstm_layers=num_layers,
                    dropout_rate=dropout,
                    vocab_size=vocab_size
                )

                if loaded_config:
                    generator.config = loaded_config

                # Now load the model and tokenizer (TextGenerator.load_model will handle missing config/tokenizer)
                generator.load_model(model_dir)
                print(f"✓ PyTorch model loaded successfully from {model_dir}")
                model_loaded = True
                break

            # Fallback: Check for legacy Keras model.h5 + config
            if model_h5.exists() and config_file.exists():
                import json
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)

                generator = TextGenerator(
                    sequence_length=loaded_config.get('sequence_length', 30),
                    embedding_dim=loaded_config.get('embedding_dim', 50),
                    lstm_units=loaded_config.get('lstm_units', 75),
                    num_lstm_layers=loaded_config.get('num_lstm_layers', 1),
                    dropout_rate=loaded_config.get('dropout_rate', 0.2),
                    vocab_size=loaded_config.get('vocab_size', None)
                )

                generator.config = loaded_config
                generator.load_model(model_dir)
                print(f"✓ Legacy Keras model loaded successfully from {model_dir}")
                model_loaded = True
                break
        
        if not model_loaded:
            print("⚠ Model not found. Please train the model first.")
            print("  Run: python train.py")
    except Exception as e:
        import traceback
        print(f"✗ Error loading model: {e}")
        print(traceback.format_exc())


@app.get("/", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        HealthResponse with status and model loaded state
    """
    is_loaded = generator is not None and generator.model is not None
    
    return HealthResponse(
        status="healthy" if is_loaded else "model_not_loaded",
        model_loaded=is_loaded
    )


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """
    Get model information and configuration.
    
    Returns:
        ModelInfo with vocabulary size, architecture details, etc.
    
    Raises:
        HTTPException: If model is not loaded
    """
    if generator is None or generator.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if generator.config is None:
        raise HTTPException(status_code=500, detail="Model config not available")
    
    return ModelInfo(
        vocabulary_size=generator.config['vocab_size'],
        sequence_length=generator.config['sequence_length'],
        embedding_dim=generator.config['embedding_dim'],
        lstm_units=generator.config['lstm_units'],
        num_layers=generator.config['num_lstm_layers'],
        is_loaded=True
    )


@app.post("/generate", response_model=GenerateResponse, tags=["Generation"])
async def generate_text(request: GenerateRequest):
    """
    Generate text from seed text.
    
    Args:
        request: GenerateRequest with seed_text, num_words, and temperature
    
    Returns:
        GenerateResponse with generated text
    
    Raises:
        HTTPException: If model is not loaded or generation fails
    """
    if generator is None or generator.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        import traceback
        print(f"\n[DEBUG] Generating text with seed: '{request.seed_text[:50]}...'")
        print(f"[DEBUG] Request: num_words={request.num_words}, temperature={request.temperature}, "
              f"top_k={request.top_k}, top_p={request.top_p}, beam_search={request.use_beam_search}")
        
        generated = generator.generate_text(
            seed_text=request.seed_text,
            num_words=request.num_words,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            use_beam_search=request.use_beam_search,
            beam_width=request.beam_width
        )
        
        # Calculate number of new words generated
        num_generated = len(generated.split()) - len(request.seed_text.split())
        
        print(f"[DEBUG] Generation successful! Generated {num_generated} words")
        
        return GenerateResponse(
            seed_text=request.seed_text,
            generated_text=generated,
            num_words_generated=num_generated,
            temperature=request.temperature
        )
    
    except Exception as e:
        error_msg = f"Error generating text: {str(e)}"
        print(f"\n[ERROR] {error_msg}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=error_msg
        )


@app.get("/visualizations/architecture", tags=["Visualizations"])
async def get_architecture():
    """
    Get model architecture diagram.
    
    Returns:
        PNG image of model architecture
    
    Raises:
        HTTPException: If visualization not found
    """
    arch_path = "visualizations/model_architecture.png"
    
    if not os.path.exists(arch_path):
        raise HTTPException(
            status_code=404,
            detail="Architecture diagram not found. Please train the model first."
        )
    
    return FileResponse(path=arch_path, media_type="image/png")


@app.get("/visualizations/training", tags=["Visualizations"])
async def get_training_history():
    """
    Get training history plot.
    
    Returns:
        PNG image of training loss and accuracy
    
    Raises:
        HTTPException: If visualization not found
    """
    history_path = "visualizations/training_history.png"
    
    if not os.path.exists(history_path):
        raise HTTPException(
            status_code=404,
            detail="Training history plot not found. Please train the model first."
        )
    
    return FileResponse(path=history_path, media_type="image/png")


@app.get("/model/summary", tags=["Model"])
async def get_model_summary():
    """
    Get detailed model summary.
    
    Returns:
        JSON with model architecture details
    
    Raises:
        HTTPException: If model is not loaded
    """
    if generator is None or generator.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        summary = generator.get_model_summary()
        return JSONResponse({"summary": summary})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", tags=["Stats"])
async def get_stats():
    """
    Get comprehensive model statistics.
    
    Returns:
        JSON with model stats, training info, etc.
    
    Raises:
        HTTPException: If model is not loaded
    """
    if generator is None or generator.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if generator.config is None:
        raise HTTPException(status_code=500, detail="Model config not available")
    
    stats = {
        "model_config": generator.config,
        "model_parameters": generator.count_params(),
        "model_layers": len(list(generator.model.named_parameters())),
        "training_history_available": generator.history is not None,
    }
    
    if generator.history is not None:
        stats["training_stats"] = {
            "final_loss": float(generator.history.history['loss'][-1]),
            "final_accuracy": float(generator.history.history['accuracy'][-1]),
            "final_val_loss": float(generator.history.history['val_loss'][-1]),
            "final_val_accuracy": float(generator.history.history['val_accuracy'][-1]),
            "epochs_trained": len(generator.history.history['loss'])
        }
    
    return stats


@app.get("/health", tags=["Health"])
async def detailed_health():
    """
    Detailed health check.
    
    Returns:
        JSON with detailed system health information
    """
    is_loaded = generator is not None and generator.model is not None
    
    health_data = {
        "status": "healthy" if is_loaded else "degraded",
        "model_loaded": is_loaded,
        "model_path_exists": os.path.exists(MODEL_PATH),
        "tokenizer_path_exists": os.path.exists(TOKENIZER_PATH),
        "config_path_exists": os.path.exists(CONFIG_PATH),
    }
    
    if is_loaded and generator.model is not None:
        health_data["model_parameters"] = int(generator.model.count_params())
        health_data["model_layers"] = len(generator.model.layers)
    
    return health_data


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint - redirect to docs."""
    return {
        "message": "RNN Text Generator API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc"
    }


# Error handling
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc)
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 70)
    print("Starting RNN Text Generator API Server")
    print("=" * 70)
    print("\nServer running at: http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 70 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
