"""
FastAPI Server for DCGAN Demo
Provides REST API and WebSocket for real-time training visualization
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import asyncio
import json
from trainer import DCGANTrainer
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="DCGAN Demo API")

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Determine compute device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    default_device = "mps"
    logger.info("Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    default_device = "cuda"
    logger.info("Using NVIDIA GPU (CUDA)")
else:
    device = torch.device("cpu")
    default_device = "cpu"
    logger.info("Using CPU")

trainer = DCGANTrainer(device=device)
training_task = None
active_connections = []


# --------------------------
#   Request Models
# --------------------------

class TrainingConfig(BaseModel):
    dataset: str = "mnist"  # mnist, fashion_mnist, cifar10
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 0.0002
    device: str = default_device  # Auto-detect GPU or CPU
    class_filter: Optional[str] = None  # CIFAR-10 class filtering


class GenerateRequest(BaseModel):
    num_images: int = 16


@app.get("/")
async def root():
    return {"message": "DCGAN Demo API is running"}


@app.get("/status")
async def get_status():
    return {
        "is_training": trainer.is_training,
        "current_epoch": trainer.current_epoch,
        "device": str(device)
    }


@app.get("/metrics")
async def get_metrics():
    return trainer.get_metrics()


@app.post("/generate")
async def generate_images(request: GenerateRequest):
    try:
        num_images = min(request.num_images, 64)

        fake_images = trainer.generate_images(num_images=num_images)
        image_b64 = trainer.images_to_base64(fake_images, nrow=int(num_images**0.5))

        return {"success": True, "image": image_b64, "num_images": num_images}

    except Exception as e:
        logger.error(f"Error generating images: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/start_training")
async def start_training(config: TrainingConfig):
    global training_task, trainer, device

    if trainer.is_training:
        return {"success": False, "message": "Training is already running"}

    try:
        # Switch device if needed
        requested_device = torch.device(config.device)
        if str(requested_device) != str(trainer.device):
            logger.info(f"Switching device: {trainer.device} -> {requested_device}")
            device = requested_device
            trainer = DCGANTrainer(device=device)

        # Build dataloader with class filter for CIFAR-10
        dataloader = trainer.get_dataloader(
            dataset_name=config.dataset,
            batch_size=config.batch_size,
            class_filter=config.class_filter
        )

        # Start training in background
        training_task = asyncio.create_task(run_training(trainer, dataloader, config.epochs))

        return {"success": True, "message": "Training started", "config": config.dict()}

    except Exception as e:
        logger.error(f"Error starting training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stop_training")
async def stop_training():
    global training_task

    if not trainer.is_training:
        return {"success": False, "message": "No active training"}

    trainer.is_training = False

    if training_task:
        training_task.cancel()
        try:
            await training_task
        except asyncio.CancelledError:
            pass

    return {"success": True, "message": "Training stopped"}


@app.post("/save_model")
async def save_model():
    if trainer.is_training:
        return {"success": False, "message": "Cannot save while training"}

    try:
        trainer.save_checkpoint("dcgan_checkpoint.pth")
        return {"success": True, "message": "Model saved", "path": "dcgan_checkpoint.pth"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/load_model")
async def load_model():
    if trainer.is_training:
        return {"success": False, "message": "Cannot load while training"}

    try:
        trainer.load_checkpoint("dcgan_checkpoint.pth")
        return {"success": True, "message": "Model loaded", "path": "dcgan_checkpoint.pth"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------
#   Training Loop
# --------------------------

async def run_training(trainer: DCGANTrainer, dataloader, num_epochs):
    trainer.is_training = True

    try:
        for epoch in range(num_epochs):
            if not trainer.is_training:
                break

            trainer.current_epoch = epoch

            for batch_idx, (real, _) in enumerate(dataloader):

                real = real.to(trainer.device)
                metrics = trainer.train_step(real)

                if batch_idx % 20 == 0:
                    await broadcast_update({
                        "type": "batch_update",
                        "epoch": epoch,
                        "batch": batch_idx,
                        "metrics": metrics
                    })

                await asyncio.sleep(0)  # Yield to event loop

            # Epoch complete — generate preview image
            try:
                sample = trainer.generate_images(16, noise=trainer.fixed_noise[:16])
                sample_b64 = trainer.images_to_base64(sample, nrow=4)
            except Exception:
                sample_b64 = None

            await broadcast_update({
                "type": "epoch_complete",
                "epoch": epoch,
                "metrics": trainer.get_metrics(),
                "sample_image": sample_b64
            })

    except asyncio.CancelledError:
        pass

    except Exception as e:
        await broadcast_update({"type": "error", "message": str(e)})

    finally:
        trainer.is_training = False
        await broadcast_update({"type": "training_complete"})


async def broadcast_update(message):
    disconnected = []
    for ws in active_connections:
        try:
            await ws.send_json(message)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        active_connections.remove(ws)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    active_connections.append(ws)

    try:
        while True:
            await asyncio.sleep(5)
            await ws.send_json({"type": "heartbeat"})
    except Exception:
        pass
    finally:
        if ws in active_connections:
            active_connections.remove(ws)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
