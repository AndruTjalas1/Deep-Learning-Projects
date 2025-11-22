"""Simplified FastAPI application for DCGAN inference only."""

from pathlib import Path
from typing import Optional
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from config import get_config
from device import get_device, get_device_info
from models import Generator

# Pydantic models for request validation
class GenerateRequest(BaseModel):
    animal_type: str = "cat"
    num_images: int = 16
    seed: Optional[int] = None

# Initialize FastAPI app
app = FastAPI(
    title="DCGAN Generation Server",
    description="API for generating cat and dog images from pre-trained DCGAN models",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
config = None
generator = None
device = None

# Model paths
MODELS_DIR = Path("./saved_models")


@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    global config, device
    
    config = get_config()
    device = get_device(config.device.use_gpu)
    
    print("Server started successfully")
    print(f"Device: {device}")
    print(f"Device info: {get_device_info()}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "DCGAN Generation Server",
        "version": "1.0.0",
        "endpoints": [
            "GET /health",
            "GET /device-info",
            "GET /available-models",
            "POST /generate",
            "POST /generate-and-save",
        ],
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "generator_loaded": generator is not None,
    }


@app.get("/device-info")
async def device_info():
    """Get device information."""
    return get_device_info()


@app.get("/available-models")
async def available_models():
    """List available pre-trained models."""
    if not MODELS_DIR.exists():
        return {"models": []}
    
    models = []
    for model_file in MODELS_DIR.glob("generator_*.pt"):
        try:
            size_mb = model_file.stat().st_size / (1024 * 1024)
            models.append({
                "filename": model_file.name,
                "size_mb": round(size_mb, 2),
            })
        except:
            pass
    
    return {"models": sorted(models, key=lambda x: x["filename"])}


@app.post("/generate")
async def generate(request: GenerateRequest):
    """
    Generate images using pre-trained generator for specific animal type.
    
    Args:
        request: GenerateRequest with animal_type, num_images, and optional seed
        
    Returns:
        List of generated images as base64 strings
    """
    global generator, device, config
    
    if config is None:
        raise HTTPException(status_code=500, detail="Server not initialized")
    
    animal_type = request.animal_type
    num_images = request.num_images
    seed = request.seed
    
    print(f"[GENERATE] Received request: animal_type={animal_type}, num_images={num_images}, seed={seed}")
    
    if num_images < 1 or num_images > 64:
        raise HTTPException(status_code=400, detail="num_images must be between 1 and 64")
    
    # Normalize animal type
    animal_type = animal_type.lower().strip()
    if animal_type not in ["cat", "dog"]:
        raise HTTPException(status_code=400, detail="animal_type must be 'cat' or 'dog'")
    
    try:
        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        # Determine model path - look for animal_type specific model first, then final
        model_names = [
            MODELS_DIR / f"generator_{animal_type}_final.pt",  # animal-specific final
            MODELS_DIR / f"generator_{animal_type}.pt",         # animal-specific
            MODELS_DIR / "generator_final.pt",                  # general final
        ]
        
        model_path = None
        for candidate_path in model_names:
            if candidate_path.exists():
                model_path = candidate_path
                break
        
        if model_path is None:
            raise HTTPException(
                status_code=400,
                detail=f"No model found for '{animal_type}'. Train a model first using: python train.py --animals {animal_type}"
            )
        
        # Load model if not already loaded or different path
        if generator is None or model_path.name != getattr(generator, "_loaded_from", ""):
            print(f"Loading generator from: {model_path}")
            
            from train import SimpleGenerator  # Import from train.py
            
            generator_new = SimpleGenerator(
                nz=config.generator.latent_dim,
                ngf=config.generator.feature_maps,
                nc=config.image.channels,
            ).to(device)
            
            generator_new.load_state_dict(torch.load(model_path, map_location=device))
            generator_new.eval()
            generator_new._loaded_from = model_path.name
            generator = generator_new
        
        # Generate images
        print(f"[GENERATE] Generating {num_images} images...")
        with torch.no_grad():
            z = torch.randn(
                num_images,
                config.generator.latent_dim,
                device=device
            )
            generated_images = generator(z)
            print(f"[GENERATE] Generated tensor shape: {generated_images.shape}")
            
            # Denormalize from [-1, 1] to [0, 1]
            generated_images = (generated_images + 1) / 2
            generated_images = torch.clamp(generated_images, 0, 1)
        
        # Convert to base64 for JSON response
        import io
        import base64
        from PIL import Image
        
        images_base64 = []
        for img in generated_images:
            # Convert to PIL Image
            img_np = (img.cpu().numpy() * 255).astype("uint8")
            img_pil = Image.fromarray(img_np.transpose(1, 2, 0))
            
            # Convert to base64
            buffer = io.BytesIO()
            img_pil.save(buffer, format="PNG")
            img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            images_base64.append(img_base64)
        
        print(f"[GENERATE] Returning {len(images_base64)} images")
        
        return {
            "success": True,
            "animal_type": animal_type,
            "num_images": len(images_base64),
            "image_size": config.image.resolution,
            "model": model_path.name,
            "images": images_base64,
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error generating images: {str(e)}")
        import traceback as tb
        tb.print_exc()
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/generate-and-save")
async def generate_and_save(
    num_images: int = 4,
    model_path: Optional[str] = None,
    output_name: str = "generated",
):
    """
    Generate images and save them to disk.
    
    Args:
        num_images: Number of images to generate
        model_path: Path to generator model
        output_name: Name for output file (without extension)
        
    Returns:
        File download response
    """
    global generator, device, config
    
    if config is None:
        raise HTTPException(status_code=500, detail="Server not initialized")
    
    try:
        from torchvision.utils import save_image
        
        # Load model
        if model_path is None:
            model_path = MODELS_DIR / "generator_final.pt"
        else:
            model_path = MODELS_DIR / model_path
        
        if not model_path.exists():
            raise HTTPException(status_code=400, detail=f"Model not found: {model_path}")
        
        generator_new = Generator(
            latent_dim=config.generator.latent_dim,
            feature_maps=config.generator.feature_maps,
            image_channels=config.image.channels,
            image_resolution=config.image.resolution,
        ).to(device)
        
        generator_new.load_state_dict(torch.load(model_path, map_location=device))
        generator_new.eval()
        
        # Generate images
        with torch.no_grad():
            z = torch.randn(
                num_images,
                config.generator.latent_dim,
                device=device
            )
            generated_images = generator_new(z)
            generated_images = (generated_images + 1) / 2
        
        # Save images
        output_dir = Path("./generated")
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"{output_name}.png"
        
        save_image(generated_images, output_file, nrow=int(num_images**0.5) or 1)
        
        return FileResponse(
            output_file,
            media_type="image/png",
            filename=f"{output_name}.png"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8090,
        log_level="info",
    )
