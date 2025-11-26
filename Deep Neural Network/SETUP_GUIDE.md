# Handwriting Recognition System - Setup Guide

Complete setup and deployment guide for the handwriting recognition project.

## Quick Start (5 minutes)

### Backend (Local)

```bash
# Navigate to backend
cd "Deep Neural Network/backend"

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
copy .env.example .env

# Run server
python app.py
```

### Frontend (Local)

```bash
# In new terminal, navigate to frontend
cd "Deep Neural Network/frontend"

# Install dependencies
npm install

# Start dev server
npm run dev
```

Visit `http://localhost:5173` in browser.

## Project Structure Overview

```
Deep Neural Network/
├── backend/
│   ├── app.py              # Flask server (main entry point)
│   ├── config.py           # Configuration
│   ├── models.py           # PyTorch models (3 algorithms)
│   ├── segmentation.py     # Character segmentation
│   ├── requirements.txt    # Python packages
│   ├── Procfile           # Heroku/Railway deployment
│   ├── railway.toml       # Railway configuration
│   ├── .env.example       # Environment template
│   ├── README.md          # Backend documentation
│   └── saved_models/      # Trained models (to be created)
│
└── frontend/
    ├── src/
    │   ├── main.jsx       # React entry
    │   ├── App.jsx        # Main component
    │   ├── api.js         # API client
    │   ├── index.css      # Global styles
    │   ├── App.css        # App styles
    │   └── components/
    │       ├── Canvas.jsx       # Drawing canvas
    │       ├── ResultDisplay.jsx # Results
    │       └── ...css files
    ├── index.html         # HTML file
    ├── vite.config.js     # Vite config
    ├── package.json       # Node dependencies
    ├── vercel.json        # Vercel config
    └── README.md          # Frontend docs
```

## Three Deep Learning Algorithms

### 1. Character CNN (Primary Model)

**File**: `backend/models.py` - `CharacterCNN` class

**Architecture**:
- Input: 28×28 grayscale images
- Conv layers: 32 → 64 → 128 filters
- Fully connected: 256 → 128 → 36 (output)
- Batch normalization, dropout for regularization
- ReLU activations, softmax output

**Purpose**: Recognize individual characters (0-9, A-Z)

**Training Data**: Kaggle handwritten characters dataset

### 2. Character Segmentation

**File**: `backend/segmentation.py`

**Technique**: Connected component analysis
- Binary thresholding
- Morphological operations (dilation/erosion)
- Component detection
- Bounding box extraction
- Left-to-right ordering

**Purpose**: Extract individual characters from continuous handwriting

**Key Functions**:
- `segment_characters()` - Main segmentation
- `standardize_character_image()` - Normalize for CNN input

### 3. Confidence Scoring (Bayesian Approach)

**File**: `backend/models.py` - `ConfidenceScorer` class

**Technique**: Monte Carlo dropout
- Multiple forward passes with dropout enabled
- Collect predictions across passes
- Compute mean and variance
- Use variance as uncertainty measure

**Purpose**: Provide confidence scores and uncertainty estimates

**Output**: Grade A-F based on confidence threshold

## API Endpoints

### Single Character Recognition

```
POST /recognize/character
Content-Type: multipart/form-data

Form data:
  - image: [PNG/JPG file]

Response:
  {
    "success": true,
    "prediction": "A",
    "grade_info": {
      "grade": "A",
      "confidence": 0.95,
      "feedback": "Excellent recognition!",
      "predicted_character": "A"
    },
    "top_predictions": [...]
  }
```

### Sentence Recognition

```
POST /recognize/sentence
Content-Type: multipart/form-data

Form data:
  - image: [PNG/JPG file]

Response:
  {
    "success": true,
    "text": "HELLO",
    "num_characters": 5,
    "average_confidence": 0.87,
    "success_rate": 0.8,
    "characters": [...]
  }
```

## Local Development Setup

### Prerequisites

- Python 3.10+
- Node.js 18+
- Git
- Virtual environment tool

### Backend Setup

1. **Create virtual environment**:
   ```bash
   cd backend
   python -m venv .venv
   .\.venv\Scripts\activate  # Windows
   source .venv/bin/activate  # macOS/Linux
   ```

2. **Install packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env as needed for local testing
   ```

4. **Download dataset** (optional for training):
   - Kaggle: https://www.kaggle.com/datasets/sujaymann/handwritten-english-characters-and-digits

5. **Run development server**:
   ```bash
   python app.py
   # Server runs on http://localhost:5000
   ```

### Frontend Setup

1. **Install dependencies**:
   ```bash
   cd frontend
   npm install
   ```

2. **Create .env.local** (optional):
   ```
   VITE_API_URL=http://localhost:5000
   ```

3. **Start dev server**:
   ```bash
   npm run dev
   # Frontend runs on http://localhost:5173
   ```

### Test the System

1. Open browser to `http://localhost:5173`
2. Try Character Recognition tab
3. Try Text Recognition tab
4. Check backend logs for API calls

## Training the Model

To train your own model:

1. **Download dataset**:
   ```bash
   # From Kaggle
   kaggle datasets download -d sujaymann/handwritten-english-characters-and-digits
   unzip to backend/data/
   ```

2. **Create training script** (example structure):
   ```python
   from models import CharacterCNN
   import torch
   
   # Load data
   # Create model
   model = CharacterCNN()
   # Train loop
   # Save model
   torch.save(model.state_dict(), 'saved_models/character_cnn.pt')
   ```

3. **Run training** (backend/data must exist first)

See analysis notebook (to be created) for complete training pipeline.

## Deployment

### Deploy Backend on Railway

1. **Push code to GitHub**

2. **Create Railway project**:
   - Go to https://railway.app
   - Create new project from GitHub repo

3. **Configure environment**:
   - `FLASK_ENV`: production
   - `CORS_ORIGINS`: https://your-vercel-app.vercel.app
   - `PORT`: 5000 (Railway auto-assigns)

4. **Railway detects** `Procfile` and deploys automatically

5. **Get backend URL**:
   - Navigate to Deployments
   - Copy public URL (e.g., `https://abc123.railway.app`)

### Deploy Frontend on Vercel

1. **Build locally** (optional):
   ```bash
   npm run build
   # Creates dist/ folder
   ```

2. **Deploy to Vercel**:
   - Go to https://vercel.com
   - Import from GitHub
   - Select `Deep Neural Network/frontend` folder
   - Set environment variable:
     - `VITE_API_URL`: [Your Railway Backend URL]
   - Deploy

3. **Get frontend URL**:
   - Vercel provides automatically (e.g., `https://abc.vercel.app`)

4. **Update backend CORS**:
   - On Railway dashboard
   - Update `CORS_ORIGINS` environment variable
   - Include Vercel frontend URL

### Testing Deployment

```bash
# Test backend health
curl https://your-railway-backend.railway.app/health

# Test API
curl -X POST \
  -F "image=@test.png" \
  https://your-railway-backend.railway.app/recognize/character
```

## Troubleshooting

### Backend Issues

**Port already in use**:
```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# macOS/Linux
lsof -i :5000
kill -9 <PID>
```

**Module not found**:
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

**CORS errors in frontend**:
- Update `CORS_ORIGINS` in backend `.env`
- Restart backend server

### Frontend Issues

**API connection refused**:
- Verify backend is running
- Check `VITE_API_URL` environment variable
- Check if backend URL is correct

**Port 5173 in use**:
```bash
# Vite will use next available port (5174, etc.)
```

**Build errors**:
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
npm run build
```

## Environment Variables

### Backend (.env)

```
FLASK_ENV=development          # or production
DEBUG=True
PORT=5000
CORS_ORIGINS=http://localhost:5173
DEVICE=cuda                    # or cpu
MODEL_CHECKPOINT_PATH=./saved_models/character_cnn.pt
LOG_LEVEL=INFO
```

### Frontend (.env.local for local, Railway for prod)

```
VITE_API_URL=http://localhost:5000  # or railway URL
```

## Performance Optimization

### Backend

- Use GPU if available (`DEVICE=cuda`)
- Cache model in memory (already done)
- Batch requests for throughput
- Monitor response times

### Frontend

- Lazy load components (already done with code splitting)
- Compress images before upload
- Use browser cache
- Minify CSS/JS (Vite does automatically)

## File Sizes & Limits

- Maximum image size: 10 MB
- Canvas resolution: Up to browser limits
- API timeout: 30 seconds
- Maximum characters per sentence: 50

## Next Steps

1. **Train models** on full Kaggle dataset
2. **Create analysis notebook** with evaluation metrics
3. **Add data augmentation** during training
4. **Implement user authentication** (optional)
5. **Add more features** (e.g., confidence threshold settings)
6. **Performance optimization** (model quantization)

## References

- **Deep Learning**: https://pytorch.org/
- **Flask**: https://flask.palletsprojects.com/
- **React**: https://react.dev/
- **Vite**: https://vitejs.dev/
- **Railway**: https://railway.app/
- **Vercel**: https://vercel.com/

## Additional Resources

- [PyTorch CNN Tutorial](https://pytorch.org/tutorials/beginner/basics/intro.html)
- [Handwriting Recognition Papers](https://arxiv.org/)
- [React Best Practices](https://react.dev/learn)
- [REST API Design](https://restfulapi.net/)

## Support & Questions

For issues:
1. Check troubleshooting section
2. Review backend README.md
3. Check frontend README.md
4. Review API response errors
5. Check browser console (F12)
6. Check backend logs

---

**Last Updated**: November 2024
**Project**: CST-435 Deep Learning - Handwriting Recognition System
