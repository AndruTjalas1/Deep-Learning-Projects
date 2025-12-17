# Handwriting Recognition System - Backend

Python Flask backend for AI-powered handwriting and text recognition using PyTorch.

## Project Overview

This system implements **three deep learning algorithms**:

1. **CNN (Convolutional Neural Network)** - Primary character/digit classifier
2. **Character Segmentation** - Extracts individual characters from continuous handwriting
3. **Confidence Scoring** - Uncertainty quantification using Bayesian approaches

## Features

- **Single Character Recognition**: Draw a character and get instant recognition with grading (A-F)
- **Continuous Text Recognition**: Write sentences and automatically segment/recognize each character
- **Confidence Scoring**: Get confidence scores and feedback for writing quality
- **Real-time Processing**: Fast inference using PyTorch
- **REST API**: Easy integration with frontend applications

## Project Structure

```
backend/
├── app.py              # Flask API server
├── config.py           # Configuration management
├── models.py           # PyTorch model definitions (3 algorithms)
├── segmentation.py     # Character segmentation logic
├── requirements.txt    # Python dependencies
├── Procfile           # Railway deployment
├── railway.toml       # Railway configuration
└── saved_models/      # Trained model checkpoints
```

## Installation

### Local Development

1. **Create virtual environment**:
```bash
python -m venv .venv
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Create .env file**:
```bash
cp .env.example .env
```

4. **Run development server**:
```bash
python app.py
```

Server will be available at `http://localhost:5000`

## API Endpoints

### 1. Single Character Recognition

**POST** `/recognize/character`

Request (multipart/form-data):
- `image`: Image file (PNG, JPG) of drawn character

Response:
```json
{
  "success": true,
  "prediction": "A",
  "grade_info": {
    "grade": "A",
    "confidence": 0.95,
    "feedback": "Excellent recognition! Very clear writing.",
    "predicted_character": "A"
  },
  "top_predictions": [
    {"character": "A", "confidence": 0.95},
    {"character": "B", "confidence": 0.03},
    {"character": "C", "confidence": 0.02}
  ]
}
```

### 2. Continuous Text Recognition

**POST** `/recognize/sentence`

Request (multipart/form-data):
- `image`: Image file of handwritten sentence

Response:
```json
{
  "success": true,
  "text": "HELLO",
  "num_characters": 5,
  "average_confidence": 0.87,
  "success_rate": 0.8,
  "characters": [
    {
      "index": 0,
      "character": "H",
      "confidence": 0.92,
      "grade": "A",
      "bbox": {"x": 10, "y": 20, "width": 45, "height": 50}
    }
  ]
}
```

### 3. Health Check

**GET** `/health`

Response:
```json
{
  "status": "healthy",
  "device": "cuda",
  "models_loaded": true
}
```

### 4. API Information

**GET** `/api/info`

Returns endpoint information and configuration details.

## Model Training

To train your own models, see the analysis notebook (will be created during project setup).

### Dataset Format

Expected dataset from Kaggle:
- Input: 28×28 grayscale images
- Output: Character labels (0-9, A-Z)
- Preprocessing: Normalization, augmentation

### Training Steps

1. Download dataset from Kaggle
2. Run preprocessing pipeline
3. Train models
4. Save checkpoints to `saved_models/`

## Deployment on Railway

1. **Connect GitHub repository** to Railway
2. **Set environment variables**:
   - `FLASK_ENV=production`
   - `CORS_ORIGINS=https://your-vercel-app.vercel.app`

3. **Deploy**:
```bash
railway deploy
```

Monitor logs:
```bash
railway logs
```

## Configuration

Edit `config.py` to customize:

- `IMG_SIZE`: Input image dimensions (default: 28)
- `NUM_CLASSES`: Number of classes (default: 36 for 0-9, A-Z)
- `MODEL_CONFIG`: Architecture hyperparameters
- `CORS_ORIGINS`: Allowed frontend domains

## Algorithm Details

### Algorithm 1: CNN Classifier
- 3 convolutional blocks with batch normalization
- Dropout for regularization
- Dense layers for classification
- Softmax output for probability distribution

### Algorithm 2: Character Segmentation
- Connected component analysis
- Bounding box detection
- Morphological operations (dilation, erosion)
- Left-to-right character ordering

### Algorithm 3: Confidence Scoring
- Monte Carlo dropout for uncertainty estimation
- Softmax probabilities as confidence
- Letter grading based on confidence thresholds

## Performance Metrics

Expected performance (on test set):
- Accuracy: ~95% for individual characters
- Segmentation Accuracy: ~90% for continuous text
- Inference Time: <100ms per character

## Error Handling

The API returns meaningful error messages:

```json
{
  "error": "No image provided"
}
```

Status codes:
- `200`: Success
- `400`: Bad request (missing parameters)
- `500`: Server error

## Environment Variables

```
FLASK_ENV          # "development" or "production"
DEBUG              # True/False for debug mode
PORT               # Server port (default: 5000)
CORS_ORIGINS       # Comma-separated allowed origins
DEVICE             # "cuda" or "cpu"
```

## Troubleshooting

**Models not loading**:
- Check `saved_models/` directory exists
- Verify model checkpoint files
- Check device compatibility (CPU vs GPU)

**CORS errors**:
- Update `CORS_ORIGINS` in `.env`
- Ensure frontend URL matches exactly

**Slow inference**:
- Consider using GPU (`DEVICE=cuda`)
- Reduce model complexity
- Batch multiple requests

## Dependencies

Key packages:
- `torch` - Deep learning framework
- `torchvision` - Image utilities
- `flask` - Web framework
- `opencv-python` - Image processing
- `numpy`, `scipy` - Numerical computing

See `requirements.txt` for complete list with versions.

## References

- PyTorch: https://pytorch.org/
- MNIST Dataset: http://yann.lecun.com/exdb/mnist/
- Handwritten Characters Dataset: https://www.kaggle.com/datasets/sujaymann/handwritten-english-characters-and-digits

## License

Course project for CST-435 Deep Learning

## Authors

- Your Name
- Your Partner's Name
