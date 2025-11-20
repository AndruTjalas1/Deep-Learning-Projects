"""Main FastAPI application for DCGAN training server."""

import asyncio
import json
from pathlib import Path
from typing import Optional, List
import threading

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from config import get_config
from device import get_device, get_device_info
from models import Generator, Discriminator, weights_init
from data_loader import create_train_loader
from trainer import DCGANTrainer


# Initialize FastAPI app
app = FastAPI(
    title="DCGAN Training Server",
    description="API for training DCGAN models on cats, dogs, and other animals",
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

# Global trainer instance
trainer: Optional[DCGANTrainer] = None
training_thread: Optional[threading.Thread] = None
config = None


@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    global config
    config = get_config()
    print("Server started successfully")
    print(f"Device info: {get_device_info()}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "DCGAN Training Server",
        "version": "1.0.0",
        "endpoints": [
            "/health",
            "/device-info",
            "/config",
            "/train/start",
            "/train/status",
            "/train/stop",
            "/generate",
            "/samples",
        ],
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "trainer_initialized": trainer is not None}


@app.get("/device-info")
async def device_info():
    """Get device information."""
    return get_device_info()


@app.get("/config")
async def get_current_config():
    """Get current configuration."""
    if config is None:
        raise HTTPException(status_code=500, detail="Configuration not loaded")
    
    return {
        "training": config.training.model_dump(),
        "image": config.image.model_dump(),
        "generator": config.generator.model_dump(),
        "discriminator": config.discriminator.model_dump(),
        "sampling": config.sampling.model_dump(),
        "data": config.data.model_dump(),
        "output": config.output.model_dump(),
    }


@app.post("/config/update")
async def update_config(
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    resolution: Optional[int] = None,
    sample_interval: Optional[int] = None,
):
    """Update configuration parameters."""
    global config
    
    if config is None:
        raise HTTPException(status_code=500, detail="Configuration not loaded")
    
    if epochs is not None:
        config.training.epochs = epochs
    if batch_size is not None:
        config.training.batch_size = batch_size
    if learning_rate is not None:
        config.training.learning_rate = learning_rate
    if resolution is not None:
        config.image.resolution = resolution
    if sample_interval is not None:
        config.sampling.sample_interval = sample_interval
    
    return {
        "message": "Configuration updated",
        "config": {
            "training": config.training.model_dump(),
            "image": config.image.model_dump(),
            "sampling": config.sampling.model_dump(),
        },
    }


@app.post("/train/start")
async def start_training(
    animal_types: Optional[List[str]] = None,
    epochs: Optional[int] = None,
):
    """Start training the DCGAN model."""
    global trainer, training_thread, config
    
    if config is None:
        raise HTTPException(status_code=500, detail="Configuration not loaded")
    
    if trainer is not None and trainer.training_started and not trainer.training_completed:
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    # Use provided epochs or config epochs
    num_epochs = epochs or config.training.epochs
    
    # Use provided animal types or default
    animal_types = animal_types or ['cats', 'dogs']
    
    try:
        # Initialize models
        device = get_device(config.device.use_gpu)
        
        generator = Generator(
            latent_dim=config.generator.latent_dim,
            feature_maps=config.generator.feature_maps,
            image_channels=config.image.channels,
            image_resolution=config.image.resolution,
        )
        
        discriminator = Discriminator(
            feature_maps=config.discriminator.feature_maps,
            image_channels=config.image.channels,
            image_resolution=config.image.resolution,
        )
        
        # Initialize weights
        generator.apply(weights_init)
        discriminator.apply(weights_init)
        
        # Create trainer
        trainer = DCGANTrainer(
            generator=generator,
            discriminator=discriminator,
            config=config.model_dump(),
            device=device,
        )
        
        # Create data loader
        train_loader = create_train_loader(
            data_dir=config.data.dataset_path,
            batch_size=config.training.batch_size,
            num_workers=config.data.num_workers,
            resolution=config.image.resolution,
            animal_types=animal_types,
        )
        
        # Start training in background thread
        def train_worker():
            trainer.train(
                train_loader=train_loader,
                num_epochs=num_epochs,
                checkpoint_interval=5,
            )
        
        training_thread = threading.Thread(target=train_worker, daemon=True)
        training_thread.start()
        
        return {
            "message": "Training started",
            "epochs": num_epochs,
            "animal_types": animal_types,
            "device": str(device),
            "image_resolution": config.image.resolution,
        }
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=f"Data not found: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting training: {str(e)}")


@app.get("/train/status")
async def training_status():
    """Get training status."""
    if trainer is None:
        return {
            "training_active": False,
            "message": "No training session initialized",
        }
    
    state = trainer.get_training_state()
    return {
        "training_active": trainer.training_started and not trainer.training_completed,
        **state,
    }


@app.post("/train/stop")
async def stop_training():
    """Stop training gracefully."""
    if trainer is None or not trainer.training_started:
        raise HTTPException(status_code=400, detail="No active training")
    
    # The trainer will save a checkpoint on interrupt
    return {
        "message": "Training stop requested",
        "current_epoch": trainer.current_epoch,
    }


@app.get("/generate")
async def generate_images(num_images: int = 16):
    """Generate random images using the current generator."""
    if trainer is None or trainer.generator is None:
        raise HTTPException(status_code=400, detail="No trained generator available")
    
    try:
        images = trainer.generate_images(num_images)
        return {
            "message": "Images generated successfully",
            "num_images": num_images,
            "image_shape": list(images.shape),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating images: {str(e)}")


@app.get("/samples")
async def list_samples():
    """List all generated samples."""
    if config is None:
        raise HTTPException(status_code=500, detail="Configuration not loaded")
    
    samples_dir = Path(config.output.samples_dir)
    
    if not samples_dir.exists():
        return {"samples": []}
    
    samples = sorted([
        str(f.relative_to(samples_dir)) for f in samples_dir.glob("*.png")
    ])
    
    return {
        "total_samples": len(samples),
        "samples": samples,
    }


@app.get("/samples/{sample_name}")
async def get_sample(sample_name: str):
    """Get a specific sample image."""
    if config is None:
        raise HTTPException(status_code=500, detail="Configuration not loaded")
    
    sample_path = Path(config.output.samples_dir) / sample_name
    
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail=f"Sample not found: {sample_name}")
    
    return FileResponse(sample_path, media_type="image/png")


@app.get("/models")
async def list_models():
    """List all saved model checkpoints."""
    if config is None:
        raise HTTPException(status_code=500, detail="Configuration not loaded")
    
    models_dir = Path(config.output.models_dir)
    
    if not models_dir.exists():
        return {"models": []}
    
    models = sorted([
        str(f.relative_to(models_dir)) for f in models_dir.glob("*.pt")
    ])
    
    return {
        "total_models": len(models),
        "models": models,
    }


@app.get("/training-metrics")
async def get_training_metrics():
    """Get training metrics (losses, etc.)."""
    if trainer is None:
        raise HTTPException(status_code=400, detail="No training session active")
    
    state = trainer.get_training_state()
    
    return {
        "current_epoch": state['current_epoch'],
        "total_epochs": state['total_epochs'],
        "g_losses": state['g_losses'],
        "d_losses": state['d_losses'],
        "latest_g_loss": state['latest_g_loss'],
        "latest_d_loss": state['latest_d_loss'],
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
