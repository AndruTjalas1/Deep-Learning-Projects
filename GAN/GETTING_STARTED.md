# DCGAN Training: Getting Started Guide

## Overview

This DCGAN system is designed to be easy to use for both beginner and advanced users. Whether you're training on a Windows machine with an NVIDIA GPU or a Mac with Apple Silicon, the setup process is straightforward.

## Step 1: Environment Setup

### Option A: Fresh Installation

```bash
# Clone the repository
cd GAN/Backend

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Option B: Using Conda (Recommended for GPU)

```bash
# Create conda environment with CUDA support (Windows/Linux)
conda create -n dcgan python=3.10 pytorch pytorch-cuda=11.8 -c pytorch -c nvidia

# Or for Mac with Apple Silicon:
conda create -n dcgan python=3.10 pytorch -c pytorch

# Activate environment
conda activate dcgan

# Install additional requirements
pip install fastapi uvicorn pydantic pyyaml pillow opencv-python numpy
```

## Step 2: Prepare Your Dataset

### Directory Structure

```
Backend/data/
├── cats/
│   ├── image_1.jpg
│   ├── image_2.png
│   └── ... (add more cat images)
└── dogs/
    ├── image_1.jpg
    ├── image_2.png
    └── ... (add more dog images)
```

### Where to Get Images

1. **Cats Dataset**: Download from:
   - Microsoft Cats and Dogs Dataset
   - Kaggle: https://kaggle.com/datasets/shaunacheng/cat-and-dog-images

2. **Dogs Dataset**: Use same sources as above

3. **Other Animals**: Create additional subdirectories like `birds/`, `bears/`, etc.

### Dataset Tips

- **Minimum images**: 100 per animal type (500+ recommended)
- **Supported formats**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`
- **No preparation needed**: System handles resizing and normalization automatically
- **Image quality**: Works best with clear, focused images of the animal

## Step 3: Configure Training

Edit `Backend/config.yaml`:

```yaml
training:
  epochs: 50           # Start with 50, increase for better quality
  batch_size: 64       # Reduce if out of VRAM (32, 16, 8)
  learning_rate: 0.0002

image:
  resolution: 64       # 64x64 for testing, 128+ for production

sampling:
  sample_interval: 100 # Save samples every 100 batches
  num_samples: 16
```

## Step 4: Start Training

### Terminal 1 - Backend Server

```bash
cd GAN/Backend
python main.py
```

You should see:
```
INFO:     Application startup complete
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Terminal 2 - Frontend (Optional but Recommended)

```bash
cd GAN/Frontend
npm install  # First time only
npm run dev
```

Opens at: `http://localhost:3000`

## Step 5: Monitor Training

### Via Web Interface (Recommended)

1. Open `http://localhost:3000`
2. Use the Configuration panel to adjust settings
3. Click "Start Training" in Training Control
4. Watch metrics update in real-time
5. View generated samples in the gallery

### Via API (Advanced)

```bash
# Check training status
curl http://localhost:8000/train/status

# Get metrics
curl http://localhost:8000/training-metrics

# List samples
curl http://localhost:8000/samples

# Check device
curl http://localhost:8000/device-info
```

## Customization

### Adding New Animal Types

1. Create directory: `Backend/data/new_animal/`
2. Add images to the directory
3. Start training: Click animals and train!

### Fine-tuning Hyperparameters

Try these configurations based on your needs:

**For Speed (Testing)**:
```yaml
epochs: 5
batch_size: 32
resolution: 64
```

**For Quality (Production)**:
```yaml
epochs: 100
batch_size: 64
resolution: 128
```

**For High-Res Images**:
```yaml
epochs: 200
batch_size: 16
resolution: 256
learning_rate: 0.0001  # Lower LR for stability
```

## Troubleshooting

### Issue: "No GPU detected"
**Solution**: The system will automatically use CPU. To force GPU:
1. Verify drivers: `nvidia-smi` (Windows/Linux)
2. Set `use_gpu: true` in config.yaml
3. Reinstall PyTorch with CUDA: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

### Issue: "CUDA out of memory"
**Solutions**:
1. Reduce batch_size (32 → 16 → 8)
2. Reduce resolution (128 → 64)
3. Reduce num_workers (4 → 2)

### Issue: "No images found"
**Solutions**:
1. Check directory path: `Backend/data/{animal_type}/`
2. Verify file extensions: must be `.jpg`, `.png`, `.bmp`, or `.gif`
3. Check permissions: images must be readable

### Issue: Frontend can't connect
**Solutions**:
1. Verify backend running: `http://localhost:8000/health`
2. Check CORS: Should see "healthy" response
3. Set API URL: `.env` file or vite config

## Next Steps

Once training is working:

1. **Deploy to Cloud**:
   - Frontend → Vercel (free tier available)
   - Backend → Railway.app or Heroku

2. **Experiment with Settings**:
   - Try different learning rates
   - Adjust batch sizes
   - Increase resolution gradually

3. **Improve Results**:
   - Add more training data
   - Train longer (100+ epochs)
   - Try different resolutions

## Performance Tips

### To Speed Up Training:
- Use smaller batch size (if VRAM allows multiple batches)
- Reduce image resolution temporarily
- Reduce number of workers (if disk I/O is bottleneck)

### To Improve Quality:
- Use larger batch size
- Train for more epochs
- Increase resolution (64 → 128 → 256)
- Use more diverse training data

### To Save Memory:
- Reduce batch_size
- Reduce resolution
- Reduce num_workers
- Use fp16 (advanced)

## Getting Help

1. Check logs: `Backend/logs/`
2. Review README.md for complete documentation
3. Check console output for errors
4. Try with smaller dataset first (test run)

## Common Questions

**Q: How long does training take?**
A: 50 epochs at 64×64 with GPU: 2-3 hours. Depends on dataset size and hardware.

**Q: What if I don't have a GPU?**
A: CPU works but is slower. Set `use_gpu: false` in config, use smaller batches.

**Q: Can I train on multiple animals?**
A: Yes! Add directories like `Backend/data/birds/`, `Backend/data/bears/`, etc.

**Q: How do I save models for later?**
A: Models auto-save to `Backend/saved_models/` every 5 epochs.

**Q: Can I pause and resume training?**
A: Stop training (saves checkpoint), then restart. It resumes from last epoch.

---

Ready to train? Start with Step 1 and follow through Step 5! 🚀
