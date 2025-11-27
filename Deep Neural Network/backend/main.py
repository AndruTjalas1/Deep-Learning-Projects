"""
FastAPI Backend for Handwriting Recognition System

Algorithm Implementation:
1. CNN Classifier - Convolutional Neural Network for character classification
2. Character Segmentation - Connected components analysis for text extraction
3. Confidence Scoring - Softmax probability-based uncertainty estimation
"""

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import io
from PIL import Image
import numpy as np
from pathlib import Path
import os
from dotenv import load_dotenv
import json
from pydantic import BaseModel

# Load environment variables
load_dotenv()

from models import CharacterCNN, ConfidenceScorer
from segmentation import batch_segment_and_standardize, preprocess_image
from config import (
    DATA_DIR, MODELS_DIR, NUM_CLASSES, IMG_SIZE, 
    CHAR_TO_IDX, IDX_TO_CHAR
)

# ============================================================
# CUDA SETUP
# ============================================================
print(f"\n{'='*70}")
print(f"PyTorch Initialization:")
print(f"  torch.__version__ = {torch.__version__}")
print(f"  torch.cuda.is_available() = {torch.cuda.is_available()}")
print(f"  torch.version.cuda = {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"  torch.cuda.get_device_name(0) = {torch.cuda.get_device_name(0)}")
print(f"{'='*70}\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✓ Using device: {device}\n")

# ============================================================
# FASTAPI SETUP
# ============================================================
app = FastAPI(
    title="Handwriting Recognition API",
    description="Deep Learning system for handwritten character recognition",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# MODEL LOADING
# ============================================================

character_classifier = None
confidence_scorer = None

def load_models():
    """Load pretrained models from disk."""
    global character_classifier, confidence_scorer
    
    try:
        # Create and load Enhanced CNN
        character_classifier = CharacterCNN(num_classes=NUM_CLASSES, dropout_rate=0.25)
        
        loaded_path = None
        
        # STRATEGY: Use epoch 63 which achieved the best validation accuracy (89.69%)
        # This epoch generalizes best across writing styles
        preferred_epochs = [50,45,40,35,30, 25, 20, 15, 10, 5, 2, 1]  # Try these in order
        for epoch_num in preferred_epochs:
            candidate = MODELS_DIR / f"epoch_{epoch_num:03d}.pt"
            if candidate.exists():
                try:
                    character_classifier.load_state_dict(torch.load(candidate, map_location=device))
                    loaded_path = candidate
                    sample_param = list(character_classifier.parameters())[0]
                    print(f"\n✓ Loaded character classifier from {candidate} (epoch {epoch_num} - better generalization)")
                    print(f"  DEBUG - First layer weight stats:")
                    print(f"    Mean: {sample_param.mean():.6f}, Std: {sample_param.std():.6f}")
                    break
                except Exception as e:
                    print(f"Warning: Error loading {candidate}: {e}")
        
        # If preferred epochs not found, find the latest epoch model
        if loaded_path is None:
            epoch_models = sorted(MODELS_DIR.glob("epoch_*.pt"), reverse=True)
            if epoch_models:
                latest_epoch_path = epoch_models[0]
                try:
                    character_classifier.load_state_dict(torch.load(latest_epoch_path, map_location=device))
                    loaded_path = latest_epoch_path
                    sample_param = list(character_classifier.parameters())[0]
                    print(f"\n✓ Loaded character classifier from {latest_epoch_path} (latest epoch)")
                    print(f"  DEBUG - First layer weight stats:")
                    print(f"    Mean: {sample_param.mean():.6f}, Std: {sample_param.std():.6f}")
                except Exception as e:
                    print(f"Warning: Error loading {latest_epoch_path}: {e}")
        
        # Fallback to character_cnn.pt if no epoch model found
        if loaded_path is None:
            model_path = MODELS_DIR / "character_cnn.pt"
            if model_path.exists():
                try:
                    character_classifier.load_state_dict(torch.load(model_path, map_location=device))
                    loaded_path = model_path
                    sample_param = list(character_classifier.parameters())[0]
                    print(f"\n✓ Loaded character classifier from {model_path}")
                    print(f"  DEBUG - First layer weight stats:")
                    print(f"    Mean: {sample_param.mean():.6f}, Std: {sample_param.std():.6f}")
                except Exception as e:
                    print(f"Warning: Error loading {model_path}: {e}")
        
        # If still no model, try character_cnn_best.pt
        if loaded_path is None:
            backup_model_path = MODELS_DIR / "character_cnn_best.pt"
            if backup_model_path.exists():
                try:
                    character_classifier.load_state_dict(torch.load(backup_model_path, map_location=device))
                    loaded_path = backup_model_path
                    sample_param = list(character_classifier.parameters())[0]
                    print(f"\n✓ Loaded character classifier from {backup_model_path} (legacy)")
                    print(f"  DEBUG - First layer weight stats:")
                    print(f"    Mean: {sample_param.mean():.6f}, Std: {sample_param.std():.6f}")
                except Exception as e:
                    print(f"Warning: Error loading {backup_model_path}: {e}")
        
        # If nothing worked, show warning
        if loaded_path is None:
            sample_param = list(character_classifier.parameters())[0]
            print(f"\n✗ WARNING: No trained model found. Using untrained model with random weights!")
            print(f"  DEBUG - First layer weight stats (SHOULD BE RANDOM):")
            print(f"    Mean: {sample_param.mean():.6f}, Std: {sample_param.std():.6f}")
        
        character_classifier.to(device)
        character_classifier.eval()
        
        # Create confidence scorer
        confidence_scorer = ConfidenceScorer(character_classifier)
        confidence_scorer.to(device)
        confidence_scorer.base_model.eval()
        
        print("Models loaded successfully")
        
    except Exception as e:
        print(f"Error loading models: {e}")
        raise


def grade_character(confidence, predicted_char):
    """
    Algorithm 3: Confidence Scoring
    Grade a character recognition based on confidence level.
    """
    if confidence >= 0.9:
        grade = "A"
        feedback = "Excellent! Clear and well-formed character."
    elif confidence >= 0.7:
        grade = "B"
        feedback = "Good! Character is recognizable."
    elif confidence >= 0.5:
        grade = "C"
        feedback = "Fair. Character is somewhat recognizable."
    elif confidence >= 0.3:
        grade = "D"
        feedback = "Poor recognition. Writing is difficult to interpret."
    else:
        grade = "F"
        feedback = "Failed to recognize. Please write more clearly."
    
    return {
        "grade": grade,
        "confidence": round(confidence, 4),
        "feedback": feedback,
        "predicted_character": predicted_char
    }


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "device": str(device),
        "models_loaded": character_classifier is not None
    }


@app.post("/recognize/character")
async def recognize_character(data: dict):
    """
    Tab 1: Single Character Recognition with Grading
    
    Endpoint for recognizing a single character drawn on canvas
    and providing a grade based on confidence.
    
    Expected POST data:
    - image: Base64 encoded image data
    
    Returns:
        JSON with prediction, grade, and top predictions
    """
    try:
        # Get base64 image from request
        image_base64 = data.get("image")
        if not image_base64:
            raise HTTPException(status_code=400, detail="No image provided")
        
        # Decode base64 to image
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
        
        import base64
        image_data = base64.b64decode(image_base64)
        canvas_image = Image.open(io.BytesIO(image_data))
        
        # Save original for debugging
        original_np = np.array(canvas_image)
        
        # Convert to RGB if needed
        if canvas_image.mode not in ['RGB', 'L']:
            canvas_image = canvas_image.convert('RGB')
        
        # Preprocess image
        processed_image = preprocess_image(canvas_image, target_size=IMG_SIZE)
        
        # Save preprocessed for debugging
        processed_for_viz = (processed_image[0] * 255).astype(np.uint8)
        
        input_tensor = torch.from_numpy(processed_image).unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            logits = character_classifier(input_tensor)
            probabilities = F.softmax(logits, dim=1)
        
        # Get prediction
        pred_idx = torch.argmax(probabilities[0]).item()
        confidence = probabilities[0][pred_idx].item()
        predicted_char = IDX_TO_CHAR[pred_idx]
        
        # Grade the character
        grade_info = grade_character(confidence, predicted_char)
        
        # Get top 3 predictions
        top_probs, top_indices = torch.topk(probabilities[0], 3)
        top_predictions = [
            {
                "character": IDX_TO_CHAR[idx.item()],
                "confidence": prob.item()
            }
            for prob, idx in zip(top_probs, top_indices)
        ]
        
        return {
            "success": True,
            "prediction": predicted_char,
            "grade_info": grade_info,
            "top_predictions": top_predictions,
            "inference_device": str(device),
            "debug": {
                "original_shape": str(original_np.shape),
                "processed_shape": str(processed_image.shape),
                "input_min_max": [float(input_tensor.min()), float(input_tensor.max())],
                "non_zero_pixels": int((input_tensor > 0.01).sum().item())
            }
        }
        
    except Exception as e:
        print(f"Error in recognize_character: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recognize/sentence")
async def recognize_sentence(data: dict):
    """
    Tab 2: Continuous Text Recognition with Segmentation
    
    Endpoint for recognizing handwritten text drawn as a sentence.
    Uses segmentation to extract individual characters and CNN for classification.
    
    Expected POST data:
    - image: Base64 encoded image data
    
    Returns:
        JSON with recognized text, per-character info, and visualization data
    """
    try:
        # Get base64 image from request
        image_base64 = data.get("image")
        if not image_base64:
            raise HTTPException(status_code=400, detail="No image provided")
        
        # Decode base64 to image
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
        
        import base64
        image_data = base64.b64decode(image_base64)
        canvas_image = Image.open(io.BytesIO(image_data))
        
        # Ensure image is in RGB mode before segmentation
        if canvas_image.mode not in ['RGB', 'L']:
            if canvas_image.mode == 'RGBA':
                canvas_image = canvas_image.convert('RGB')
            else:
                canvas_image = canvas_image.convert('RGB')
        
        # Segment characters
        try:
            segmented_images, bboxes = batch_segment_and_standardize(
                canvas_image, 
                target_size=IMG_SIZE
            )
        except Exception as seg_error:
            print(f"Segmentation error: {seg_error}")
            return {
                "success": True,
                "text": "",
                "message": f"Segmentation error: {str(seg_error)}",
                "characters": [],
                "error": str(seg_error)
            }
        
        if len(segmented_images) == 0:
            return {
                "success": True,
                "text": "",
                "message": "No characters detected. Please write more clearly on the canvas.",
                "characters": [],
                "num_characters": 0,
                "average_confidence": 0,
                "success_rate": 0
            }
        
        # Convert to tensor
        try:
            input_tensor = torch.from_numpy(segmented_images).to(device)
        except Exception as tensor_error:
            raise HTTPException(status_code=500, detail=f"Tensor conversion error: {str(tensor_error)}")
        
        # Run inference on all characters
        try:
            with torch.no_grad():
                logits = character_classifier(input_tensor)
                probabilities = F.softmax(logits, dim=1)
        except Exception as inference_error:
            raise HTTPException(status_code=500, detail=f"Model inference error: {str(inference_error)}")
        
        # Extract predictions
        pred_indices = torch.argmax(probabilities, dim=1)
        confidences = torch.max(probabilities, dim=1)[0]
        
        # DEBUG: Show predictions for sentence
        print(f"\nDEBUG recognize_sentence:")
        print(f"  Total characters detected: {len(pred_indices)}")
        print(f"  Predictions: {[IDX_TO_CHAR[idx.item()] for idx in pred_indices[:10]]}")
        print(f"  Confidences: {[f'{c:.4f}' for c in confidences[:10].tolist()]}")
        
        recognized_text = ""
        character_results = []
        
        for pred_idx, confidence in zip(pred_indices, confidences):
            pred_char = IDX_TO_CHAR[pred_idx.item()]
            conf_value = confidence.item()
            recognized_text += pred_char
            
            # Grade each character
            grade_info = grade_character(conf_value, pred_char)
            
            character_results.append({
                "character": pred_char,
                "confidence": round(conf_value, 4),
                "grade": grade_info["grade"]
            })
        
        # Calculate statistics
        if confidences.numel() > 0:
            avg_confidence = confidences.mean().item()
            success_count = (confidences > 0.7).sum().item()
            success_rate = (success_count / len(confidences)) * 100
        else:
            avg_confidence = 0
            success_rate = 0
        
        return {
            "success": True,
            "text": recognized_text,
            "characters": character_results,
            "num_characters": len(character_results),
            "average_confidence": round(avg_confidence, 4),
            "success_rate": round(success_rate, 2),
            "message": f"Recognized {len(character_results)} characters"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in recognize_sentence: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Handwriting Recognition API",
        "version": "1.0.0",
        "device": str(device),
        "endpoints": {
            "health": "/health (GET)",
            "character": "/recognize/character (POST)",
            "sentence": "/recognize/sentence (POST)"
        }
    }


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    load_models()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
