# DCGAN Training System

A complete Deep Convolutional Generative Adversarial Network (DCGAN) setup for generating images of cats, dogs, and other animals. Designed for distributed training on both Windows (NVIDIA GPU) and Mac (Apple Silicon), with a real-time React frontend for monitoring training progress.

## Features

✨ **Multi-Animal Support**: Train models for cats, dogs, and easily add more animals
🎨 **Real-Time Monitoring**: Live training metrics and sample visualization
🚀 **GPU Acceleration**: Automatic support for NVIDIA CUDA, Apple MPS, and CPU fallback
⚙️ **Easy Configuration**: YAML-based config with epochs, resolution, batch size customization
📊 **Training Visualizations**: Loss tracking and sample generation during training
🔧 **Modular Architecture**: Clean, maintainable Python code following modern best practices
🌐 **Full Stack**: FastAPI backend + React frontend ready for deployment

## Project Structure

```
GAN/
├── Backend/           # FastAPI server & training logic
│   ├── config.yaml    # Hyperparameters (epochs, resolution, etc.)
│   ├── config.py      # Configuration management
│   ├── device.py      # GPU/CPU device handling
│   ├── data_loader.py # Dataset utilities
│   ├── models.py      # DCGAN architecture
│   ├── trainer.py     # Training loop with sampling
│   ├── main.py        # FastAPI application
│   ├── requirements.txt
│   ├── data/          # Training data directory
│   │   ├── cats/
│   │   └── dogs/
│   ├── samples/       # Generated samples during training
│   ├── saved_models/  # Model checkpoints
│   └── logs/          # Training logs
│
└── Frontend/          # React + Vite application
    ├── package.json
    ├── vite.config.js
    ├── vercel.json
    ├── index.html
    └── src/
        ├── main.jsx
        ├── App.jsx
        ├── App.css
        ├── index.css
        ├── api.js
        └── components/
            ├── ConfigPanel.jsx      # Hyperparameter control
            ├── TrainingControl.jsx  # Start/stop training
            ├── MetricsDisplay.jsx   # Loss visualization
            └── GalleryView.jsx      # Sample gallery
```

## Quick Start

### Prerequisites

- **Windows**: Python 3.10+, NVIDIA GPU with CUDA (or CPU)
- **Mac**: Python 3.10+, Apple Silicon (or Intel with MPS support)
- Node.js 16+ for frontend

### Backend Setup

1. **Install Python dependencies**:
   ```bash
   cd Backend
   pip install -r requirements.txt
   ```

2. **Prepare dataset**:
   ```
   Backend/data/
   ├── cats/
   │   ├── cat1.jpg
   │   ├── cat2.jpg
   │   └── ...
   └── dogs/
       ├── dog1.jpg
       ├── dog2.jpg
       └── ...
   ```

   The system accepts `.jpg`, `.png`, `.bmp`, and `.gif` files. Images will be automatically:
   - Resized to configured resolution (default 64×64)
   - Center-cropped
   - Normalized (pixel values: -1 to 1)

3. **Configure hyperparameters** (`config.yaml`):
   ```yaml
   training:
     epochs: 50           # Number of training epochs
     batch_size: 64       # Batch size
     learning_rate: 0.0002
   
   image:
     resolution: 64       # 64, 128, or 256
     channels: 3
   
   sampling:
     sample_interval: 100 # Generate samples every N batches
     num_samples: 16      # Images per sample grid
   ```

4. **Start backend server**:
   ```bash
   python main.py
   ```
   Server runs on `http://localhost:8000`

### Frontend Setup

1. **Install dependencies**:
   ```bash
   cd Frontend
   npm install
   ```

2. **Start development server**:
   ```bash
   npm run dev
   ```
   Opens at `http://localhost:3000`

3. **Build for production**:
   ```bash
   npm run build
   ```

## Configuration Guide

### Key Parameters in `config.yaml`

| Parameter | Range | Notes |
|-----------|-------|-------|
| **epochs** | 1-1000 | More epochs = better quality but longer training |
| **batch_size** | 8-512 | Larger batches need more VRAM |
| **learning_rate** | 0.00001-0.1 | Default 0.0002 works well |
| **resolution** | 64/128/256 | Higher = better quality, more resources |
| **sample_interval** | 10-1000 | Generate samples every N batches |

### Recommended Configurations

**Fast Testing** (for development):
```yaml
training:
  epochs: 5
  batch_size: 32
image:
  resolution: 64
sampling:
  sample_interval: 50
```

**Production** (for quality results):
```yaml
training:
  epochs: 100
  batch_size: 64
image:
  resolution: 128
sampling:
  sample_interval: 100
```

**High Resolution**:
```yaml
training:
  epochs: 200
  batch_size: 32
image:
  resolution: 256
sampling:
  sample_interval: 200
```

## API Endpoints

### Training Control

- `POST /train/start` - Start training
  - Params: `animal_types` (list), `epochs` (int)
  
- `GET /train/status` - Get current training status
  
- `POST /train/stop` - Stop training gracefully

### Configuration

- `GET /config` - Get current configuration
- `POST /config/update` - Update parameters (epochs, batch_size, resolution, etc.)

### Monitoring

- `GET /training-metrics` - Get loss values and training progress
- `GET /samples` - List all generated samples
- `GET /samples/{name}` - Get specific sample image
- `GET /models` - List saved model checkpoints

### Utilities

- `GET /health` - Health check
- `GET /device-info` - Device information (GPU/CPU details)
- `GET /generate` - Generate images with current model

## Usage Examples

### Using the Web Interface

1. **Configure Training**:
   - Adjust epochs, batch size, resolution in the Configuration panel
   - Click "Update Configuration"

2. **Start Training**:
   - Select animal types (cats, dogs, etc.)
   - Set number of epochs
   - Click "Start Training"

3. **Monitor Progress**:
   - Watch real-time loss metrics in the chart
   - View generated samples in the gallery
   - Check current epoch and loss values

4. **Stop Training**:
   - Click "Stop Training" button
   - Current checkpoint is saved automatically

### Using the API Directly

```bash
# Start training
curl -X POST http://localhost:8000/train/start \
  -H "Content-Type: application/json" \
  -d '{"animal_types": ["cats", "dogs"], "epochs": 50}'

# Check status
curl http://localhost:8000/train/status

# Get metrics
curl http://localhost:8000/training-metrics

# Get device info
curl http://localhost:8000/device-info
```

## Advanced Features

### Multi-GPU Support

The system automatically detects and uses available GPUs:
- **Windows/Linux**: NVIDIA CUDA (auto-detected)
- **Mac**: Apple Metal Performance Shaders (MPS) or CPU
- **Fallback**: CPU if no GPU available

### Sample Generation

Generated samples are saved during training:
- Location: `Backend/samples/`
- Format: PNG grids with configurable number of images
- Named: `epoch_XXX_batch_XXXXX.png`

### Model Checkpoints

Training creates checkpoints:
- Location: `Backend/saved_models/`
- Frequency: Every 5 epochs (configurable)
- Contains: Generator, Discriminator, Optimizers, Loss history

### Training Logs

Detailed logs available in `Backend/logs/` directory with training metrics and system info.

## Deployment

### Vercel Frontend

1. Connect your GitHub repo to Vercel
2. Configure build command: `npm run build`
3. Configure output directory: `dist`
4. Set environment variable:
   ```
   VITE_API_URL=https://your-backend-api.com
   ```

### Railway Backend

1. Create Railway project
2. Connect to GitHub repo
3. Set Python version: 3.10
4. Configure start command:
   ```
   python main.py
   ```
5. Expose port: 8000
6. Add environment variable:
   ```
   PYTHONUNBUFFERED=1
   ```

## Troubleshooting

### "No GPU detected"
- Verify NVIDIA drivers installed (Windows/Linux)
- Check Apple Silicon Mac with `python -c "import torch; print(torch.backends.mps.is_available())"`
- Set `use_gpu: false` in config to use CPU

### "No images found"
- Verify dataset structure: `Backend/data/{cats,dogs}/`
- Check file extensions: `.jpg`, `.png`, `.bmp`, `.gif`
- Ensure at least a few images in each directory

### "CUDA out of memory"
- Reduce batch size in config.yaml
- Reduce image resolution (64 instead of 128)
- Use fewer workers: `num_workers: 2`

### Frontend can't connect to backend
- Ensure backend running on `http://localhost:8000`
- Check VITE_API_URL environment variable
- Verify CORS is enabled (FastAPI middleware)

## Future Enhancements

- [ ] Support for additional animal types (birds, bears, etc.)
- [ ] Transfer learning from pretrained models
- [ ] Conditional GAN (cGAN) for animal-specific generation
- [ ] Progressive GAN for higher resolution (512×512+)
- [ ] Model comparison dashboard
- [ ] Batch image download
- [ ] Custom loss metrics and discriminator improvements

## Performance Metrics

**Typical Training Times** (with GPU):

| Resolution | Batch Size | 50 Epochs | 100 Epochs |
|------------|-----------|-----------|-----------|
| 64×64      | 64        | 2-3 hrs   | 4-6 hrs   |
| 128×128    | 32        | 5-7 hrs   | 10-14 hrs |
| 256×256    | 16        | 12-16 hrs | 24-32 hrs |

**Memory Usage**:
- Batch size 64, resolution 64×64: ~4 GB VRAM
- Batch size 32, resolution 128×128: ~8 GB VRAM
- Batch size 16, resolution 256×256: ~10 GB VRAM

## Contributing

When making changes:
1. Follow PEP 8 style guidelines (Python)
2. Use ES6+ features (JavaScript)
3. Add docstrings to new functions
4. Test on both Windows and Mac
5. Verify GPU detection works correctly

## License

This project is open source. See LICENSE file for details.

## Support

For issues or questions:
1. Check the Troubleshooting section above
2. Review FastAPI logs: `Backend/logs/`
3. Check browser console for frontend errors
4. Verify device info: GET `/device-info`

---

**Happy Training! 🚀**
