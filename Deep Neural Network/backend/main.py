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
    CHAR_TO_IDX, IDX_TO_CHAR,
    DIGIT_CLASSES, UPPERCASE_CLASSES, LOWERCASE_CLASSES,
    DIGIT_IDX_TO_CHAR, UPPERCASE_IDX_TO_CHAR, LOWERCASE_IDX_TO_CHAR
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

# Specialist models for each character type
specialist_models = {
    'digit': None,
    'uppercase': None,
    'lowercase': None
}
character_type_classifier = None  # NEW: Light classifier for auto-routing
confidence_scorer = None

def load_specialist_models():
    """Load specialist models for each character type."""
    global specialist_models, character_type_classifier, confidence_scorer
    
    try:
        print("\nLoading Specialist Models...")
        
        # Load each specialist model
        for model_type in ['digit', 'uppercase', 'lowercase']:
            if model_type == 'digit':
                num_classes = len(DIGIT_CLASSES)
            elif model_type == 'uppercase':
                num_classes = len(UPPERCASE_CLASSES)
            else:
                num_classes = len(LOWERCASE_CLASSES)
            
            # Create model
            model = CharacterCNN(num_classes=num_classes, dropout_rate=0.25)
            
            # Try to load best model
            model_path = MODELS_DIR / f"best_{model_type}_model.pt"
            if model_path.exists():
                try:
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    specialist_models[model_type] = model.to(device).eval()
                    print(f"✓ Loaded {model_type} specialist model from {model_path}")
                except Exception as e:
                    print(f"Warning: Failed to load {model_type} model: {e}")
                    print(f"  Using untrained model for {model_type}")
                    specialist_models[model_type] = model.to(device).eval()
            else:
                print(f"⚠ No trained model found for {model_type} at {model_path}")
                print(f"  Using untrained model - please train first!")
                specialist_models[model_type] = model.to(device).eval()
        
        # NEW: Load character type classifier
        print("\nLoading Character Type Classifier...")
        from models import LightCharacterTypeClassifier
        character_type_classifier = LightCharacterTypeClassifier(num_classes=3)
        classifier_path = MODELS_DIR / "character_type_classifier.pt"
        
        if classifier_path.exists():
            try:
                character_type_classifier.load_state_dict(torch.load(classifier_path, map_location=device))
                character_type_classifier = character_type_classifier.to(device).eval()
                print(f"✓ Loaded character type classifier from {classifier_path}")
            except Exception as e:
                print(f"Warning: Failed to load character type classifier: {e}")
                print(f"  Using untrained classifier - please train first!")
                character_type_classifier = character_type_classifier.to(device).eval()
        else:
            print(f"⚠ No trained classifier found at {classifier_path}")
            print(f"  Run: python train_character_type_classifier.py")
            character_type_classifier = character_type_classifier.to(device).eval()
        
        # Create confidence scorer using digit model as reference
        if specialist_models['digit'] is not None:
            confidence_scorer = ConfidenceScorer(specialist_models['digit'])
            confidence_scorer.to(device)
            confidence_scorer.base_model.eval()
        
        print("✓ All specialist models and classifier loaded successfully\n")
        
    except Exception as e:
        print(f"Error loading specialist models: {e}")
        raise


# Keep backward compatibility
character_classifier = None
def load_models():
    """Backward compatibility wrapper - loads specialist models."""
    global character_classifier
    load_specialist_models()
    character_classifier = specialist_models['digit']  # Default to digit model


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


@app.post("/recognize/character/specialist")
async def recognize_character_specialist(data: dict):
    """
    SPECIALIST MODEL ENDPOINT: Single Character Recognition with User-Selected Type
    
    User selects character type: "digit", "uppercase", or "lowercase"
    Routes to the appropriate specialist model.
    
    Expected POST data:
    - image: Base64 encoded image data
    - character_type: "digit", "uppercase", or "lowercase"
    
    Returns:
        JSON with prediction, grade, confidence, and top predictions from specialist model
    """
    try:
        # Get parameters
        image_base64 = data.get("image")
        character_type = data.get("character_type", "digit").lower()
        
        if not image_base64:
            raise HTTPException(status_code=400, detail="No image provided")
        
        if character_type not in ['digit', 'uppercase', 'lowercase']:
            raise HTTPException(status_code=400, detail="Invalid character_type. Must be 'digit', 'uppercase', or 'lowercase'")
        
        # Get appropriate model and class mappings
        model = specialist_models[character_type]
        if model is None:
            raise HTTPException(status_code=500, detail=f"Specialist model for {character_type} not loaded")
        
        if character_type == 'digit':
            idx_to_char = DIGIT_IDX_TO_CHAR
        elif character_type == 'uppercase':
            idx_to_char = UPPERCASE_IDX_TO_CHAR
        else:
            idx_to_char = LOWERCASE_IDX_TO_CHAR
        
        # Decode base64 to image
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
        
        import base64
        image_data = base64.b64decode(image_base64)
        canvas_image = Image.open(io.BytesIO(image_data))
        
        # Preprocess image
        processed_image = preprocess_image(canvas_image, target_size=IMG_SIZE)
        input_tensor = torch.from_numpy(processed_image).unsqueeze(0).to(device)
        
        # Run inference with specialist model
        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = F.softmax(logits, dim=1)
        
        # Get prediction
        pred_idx = torch.argmax(probabilities[0]).item()
        confidence = probabilities[0][pred_idx].item()
        predicted_char = idx_to_char[pred_idx]
        
        # Grade the character
        grade_info = grade_character(confidence, predicted_char)
        
        # Get top 3 predictions
        top_probs, top_indices = torch.topk(probabilities[0], min(3, len(idx_to_char)))
        top_predictions = [
            {
                "character": idx_to_char[idx.item()],
                "confidence": prob.item(),
                "rank": i + 1
            }
            for i, (prob, idx) in enumerate(zip(top_probs, top_indices))
        ]
        
        return {
            "success": True,
            "character_type": character_type,
            "predicted_character": predicted_char,
            "confidence": round(confidence, 4),
            "grade_info": grade_info,
            "top_predictions": top_predictions,
            "inference_device": str(device)
        }
        
    except Exception as e:
        print(f"Error in recognize_character_specialist: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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


@app.post("/recognize/text")
async def recognize_text(data: dict):
    """
    OCR ENDPOINT: Handwritten Text Recognition with Auto-Routing
    
    Recognizes mixed-case handwritten text automatically using:
    1. Character segmentation to extract individual characters
    2. Light classifier to detect type (digit/uppercase/lowercase)
    3. Specialist models to recognize each character
    
    Expected POST data:
    - image: Base64 encoded image of handwritten text
    
    Returns:
        JSON with:
        - recognized_text: Full recognized text string
        - character_details: Per-character predictions and confidence
        - overall_confidence: Average confidence across all characters
    """
    try:
        import base64
        
        # Get image
        image_base64 = data.get("image")
        if not image_base64:
            raise HTTPException(status_code=400, detail="No image provided")
        
        # Decode base64
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
        
        image_data = base64.b64decode(image_base64)
        text_image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if needed
        if text_image.mode not in ['RGB', 'L']:
            text_image = text_image.convert('RGB')
        
        # Segment the image into individual characters
        char_batch, bounding_boxes = batch_segment_and_standardize(text_image, target_size=IMG_SIZE)
        
        if char_batch.shape[0] == 0:
            return {
                "success": False,
                "error": "No characters detected in image",
                "recognized_text": "",
                "character_details": []
            }
        
        # Process each segmented character
        recognized_text = ""
        character_details = []
        confidences = []
        
        for idx, char_image in enumerate(char_batch):
            try:
                # Convert to tensor
                if isinstance(char_image, np.ndarray):
                    char_tensor = torch.from_numpy(char_image).unsqueeze(0).to(device)
                else:
                    # Already a tensor
                    char_tensor = char_image.unsqueeze(0).to(device)
                
                # Step 1: Classify character type using light classifier
                with torch.no_grad():
                    type_logits = character_type_classifier(char_tensor)
                    type_probs = F.softmax(type_logits, dim=1)
                
                char_type_idx = torch.argmax(type_probs[0]).item()
                char_type_names = ['digit', 'uppercase', 'lowercase']
                detected_type = char_type_names[char_type_idx]
                type_confidence = type_probs[0][char_type_idx].item()
                
                # Step 2: Route to appropriate specialist model
                if detected_type == 'digit':
                    specialist_model = specialist_models['digit']
                    idx_to_char_map = DIGIT_IDX_TO_CHAR
                elif detected_type == 'uppercase':
                    specialist_model = specialist_models['uppercase']
                    idx_to_char_map = UPPERCASE_IDX_TO_CHAR
                else:
                    specialist_model = specialist_models['lowercase']
                    idx_to_char_map = LOWERCASE_IDX_TO_CHAR
                
                # Step 3: Get character prediction from specialist model
                with torch.no_grad():
                    char_logits = specialist_model(char_tensor)
                    char_probs = F.softmax(char_logits, dim=1)
                
                char_pred_idx = torch.argmax(char_probs[0]).item()
                char_confidence = char_probs[0][char_pred_idx].item()
                predicted_char = idx_to_char_map[char_pred_idx]
                
                # Step 4: Get top 3 alternatives
                top_probs, top_indices = torch.topk(char_probs[0], min(3, len(idx_to_char_map)))
                alternatives = [
                    {
                        "character": idx_to_char_map[idx.item()],
                        "confidence": prob.item(),
                        "rank": i + 1
                    }
                    for i, (prob, idx) in enumerate(zip(top_probs, top_indices))
                ]
                
                # Add to results
                recognized_text += predicted_char
                confidences.append(char_confidence)
                
                character_details.append({
                    "position": idx,
                    "predicted_character": predicted_char,
                    "confidence": round(char_confidence, 4),
                    "detected_type": detected_type,
                    "type_confidence": round(type_confidence, 4),
                    "alternatives": alternatives
                })
                
            except Exception as char_error:
                print(f"Error processing character {idx}: {char_error}")
                character_details.append({
                    "position": idx,
                    "error": str(char_error),
                    "predicted_character": "?"
                })
        
        # Calculate overall confidence
        overall_confidence = np.mean(confidences) if confidences else 0.0
        
        return {
            "success": True,
            "recognized_text": recognized_text,
            "character_count": len(recognized_text),
            "character_details": character_details,
            "overall_confidence": round(overall_confidence, 4),
            "inference_device": str(device)
        }
        
    except Exception as e:
        print(f"Error in recognize_text: {e}")
        import traceback
        traceback.print_exc()
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
        
        # Process each character with character-type classifier + specialist routing
        recognized_text = ""
        character_results = []
        confidences = []
        type_predictions = []  # Track what type each was classified as
        
        for char_idx, char_image in enumerate(segmented_images):
            try:
                # Prepare single character tensor
                char_tensor = torch.from_numpy(char_image).unsqueeze(0).to(device)
                
                # Step 1: Classify character type using light classifier
                with torch.no_grad():
                    type_logits = character_type_classifier(char_tensor)
                    type_probs = F.softmax(type_logits, dim=1)
                
                char_type_idx = torch.argmax(type_probs[0]).item()
                char_type_names = ['digit', 'uppercase', 'lowercase']
                detected_type = char_type_names[char_type_idx]
                type_confidence = type_probs[0][char_type_idx].item()
                
                # Step 2: Route to appropriate specialist model
                if detected_type == 'digit':
                    specialist_model = specialist_models['digit']
                    idx_to_char_map = DIGIT_IDX_TO_CHAR
                elif detected_type == 'uppercase':
                    specialist_model = specialist_models['uppercase']
                    idx_to_char_map = UPPERCASE_IDX_TO_CHAR
                else:  # lowercase
                    specialist_model = specialist_models['lowercase']
                    idx_to_char_map = LOWERCASE_IDX_TO_CHAR
                
                # Step 3: Get character prediction from specialist model
                with torch.no_grad():
                    char_logits = specialist_model(char_tensor)
                    char_probs = F.softmax(char_logits, dim=1)
                
                char_pred_idx = torch.argmax(char_probs[0]).item()
                char_confidence = char_probs[0][char_pred_idx].item()
                predicted_char = idx_to_char_map[char_pred_idx]
                
                recognized_text += predicted_char
                confidences.append(char_confidence)
                type_predictions.append(detected_type)
                
                character_results.append({
                    "character": predicted_char,
                    "confidence": round(char_confidence, 4),
                    "detected_type": detected_type,
                    "type_confidence": round(type_confidence, 4)
                })
                
            except Exception as char_error:
                print(f"Error processing character {char_idx}: {char_error}")
                character_results.append({
                    "character": "?",
                    "error": str(char_error)
                })
        
        # DEBUG: Show predictions for sentence
        print(f"\nDEBUG recognize_sentence:")
        print(f"  Total characters detected: {len(recognized_text)}")
        print(f"  Predictions: {list(recognized_text)}")
        print(f"  Types detected: {type_predictions}")
        print(f"  Confidences: {[f'{c:.4f}' for c in confidences]}")
        
        # Calculate statistics
        if len(confidences) > 0:
            avg_confidence = np.mean(confidences)
            success_count = sum(1 for c in confidences if c > 0.7)
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
