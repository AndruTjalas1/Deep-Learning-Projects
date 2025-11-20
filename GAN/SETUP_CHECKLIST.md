# DCGAN System - Complete Setup Checklist

Use this checklist to ensure your DCGAN system is properly configured and ready for training.

## ✅ Environment Setup

- [ ] Python 3.10+ installed
  ```bash
  python --version
  ```

- [ ] Virtual environment created
  ```bash
  cd Backend
  python -m venv venv
  ```

- [ ] Virtual environment activated
  - Windows: `venv\Scripts\activate`
  - Mac/Linux: `source venv/bin/activate`

- [ ] Dependencies installed
  ```bash
  pip install -r requirements.txt
  ```

- [ ] PyTorch installed (CPU is fine to start)
  ```bash
  python -c "import torch; print(torch.__version__)"
  ```

- [ ] *(Optional)* CUDA-enabled PyTorch installed (for speed)
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```

## 📦 Project Structure

- [ ] Backend directory structure created
  ```
  Backend/
  ├── data/
  │   ├── cats/
  │   ├── dogs/
  │   └── (other animals)
  ├── samples/
  ├── saved_models/
  ├── logs/
  ├── config.yaml
  ├── main.py
  ├── models.py
  ├── trainer.py
  └── ...
  ```

- [ ] Frontend directory structure created
  ```
  Frontend/
  ├── src/
  │   ├── components/
  │   ├── api.js
  │   ├── App.jsx
  │   └── main.jsx
  ├── package.json
  ├── vite.config.js
  └── index.html
  ```

## 🖼️ Dataset Preparation

- [ ] Created `Backend/data/cats/` directory
- [ ] Created `Backend/data/dogs/` directory
- [ ] Added cat images (minimum 50, recommended 500+)
- [ ] Added dog images (minimum 50, recommended 500+)
- [ ] Verified image formats (`.jpg`, `.png`, `.bmp`, `.gif`)
- [ ] Images are clear and focused on the animal
- [ ] *(Optional)* Added other animal types (birds, bears, etc.)

## ⚙️ Configuration

- [ ] Reviewed `Backend/config.yaml`
- [ ] Adjusted hyperparameters as needed
  - [ ] `epochs`: Set to desired value (50-100 recommended)
  - [ ] `batch_size`: Set based on GPU memory (64 for GPU, 32 for CPU)
  - [ ] `resolution`: Set to 64, 128, or 256
  - [ ] `learning_rate`: Keep at 0.0002 or adjust cautiously

- [ ] Verified dataset path: `Backend/data/`
- [ ] Verified output directories configured
  - [ ] `samples_dir`: `Backend/samples/`
  - [ ] `models_dir`: `Backend/saved_models/`
  - [ ] `logs_dir`: `Backend/logs/`

## 🚀 Backend Server

- [ ] Backend dependencies verified
  ```bash
  cd Backend
  python -m pip list | grep -E "fastapi|torch|pydantic"
  ```

- [ ] Device detection working
  ```bash
  python -c "from device import get_device_info; print(get_device_info())"
  ```

- [ ] Backend server starts successfully
  ```bash
  python main.py
  ```
  Expected output: `Uvicorn running on http://0.0.0.0:8000`

- [ ] API endpoints responding
  ```bash
  curl http://localhost:8000/health
  ```
  Expected: `{"status":"healthy","trainer_initialized":false}`

## 🌐 Frontend Setup

- [ ] Node.js 16+ installed
  ```bash
  node --version
  npm --version
  ```

- [ ] Frontend dependencies installed
  ```bash
  cd Frontend
  npm install
  ```

- [ ] Environment file created (optional)
  ```bash
  cp .env.example .env
  ```

- [ ] Frontend starts successfully
  ```bash
  npm run dev
  ```
  Expected: Opens at `http://localhost:3000`

- [ ] Frontend can reach backend
  ```bash
  curl http://localhost:8000/device-info
  ```

## 🧠 Training Verification

- [ ] Can start training from web interface
  - [ ] Configuration panel accessible
  - [ ] Animal types selectable
  - [ ] "Start Training" button clickable

- [ ] Can view training progress
  - [ ] Status updates in real-time
  - [ ] Loss metrics displayed
  - [ ] Progress bar moving

- [ ] Samples being generated
  - [ ] Gallery showing images
  - [ ] Samples folder populating
  - [ ] Images updating during training

## 📊 Monitoring

- [ ] Can view training metrics
  ```bash
  curl http://localhost:8000/training-metrics
  ```

- [ ] Can access device information
  ```bash
  curl http://localhost:8000/device-info
  ```

- [ ] Logs being created
  - [ ] Check `Backend/logs/` directory

- [ ] Samples being saved
  - [ ] Check `Backend/samples/` directory

- [ ] Models being checkpointed
  - [ ] Check `Backend/saved_models/` directory

## 🚨 Troubleshooting

### If Backend Won't Start

- [ ] Check port 8000 is available
  ```bash
  netstat -ano | findstr :8000  # Windows
  lsof -i :8000  # Mac/Linux
  ```

- [ ] Verify all dependencies installed
  ```bash
  pip install -r requirements.txt
  ```

- [ ] Check for Python errors
  ```bash
  python main.py  # Look for error messages
  ```

### If Frontend Won't Start

- [ ] Check port 3000 is available
- [ ] Clear node_modules and reinstall
  ```bash
  rm -rf node_modules package-lock.json
  npm install
  ```

- [ ] Check Node version compatibility

### If Training Won't Start

- [ ] Verify dataset exists
  ```bash
  ls Backend/data/cats/
  ls Backend/data/dogs/
  ```

- [ ] Check file permissions
- [ ] Verify image formats are supported

### If No GPU Detected

- [ ] Check NVIDIA drivers installed
  ```bash
  nvidia-smi
  ```

- [ ] Install CUDA-enabled PyTorch
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```

- [ ] Verify CUDA installation
  ```bash
  python -c "import torch; print(torch.cuda.is_available())"
  ```

## 🎯 Quick Start Commands

Copy-paste ready commands:

```bash
# Setup (one time)
cd GAN/Backend
python -m venv venv
venv\Scripts\activate  # Windows
# or: source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt

# Start training (terminal 1)
cd Backend
python main.py

# Start frontend (terminal 2)
cd Frontend
npm install  # First time only
npm run dev
```

Then open: `http://localhost:3000`

## 📈 Performance Targets

- [ ] Backend responds < 100ms
- [ ] Frontend loads < 2 seconds
- [ ] Training generates samples every 1-5 minutes
- [ ] GPU memory usage < 80% of capacity
- [ ] CPU usage < 50% (if GPU available)

## 🚢 Deployment Readiness

- [ ] Frontend build succeeds
  ```bash
  npm run build
  ```

- [ ] Frontend dist folder exists
  ```bash
  ls Frontend/dist/
  ```

- [ ] Backend handles CORS correctly
- [ ] Environment variables configured
- [ ] Dataset strategy for production planned
- [ ] Model persistence tested

## ✨ Advanced (Optional)

- [ ] GPU setup optimized
- [ ] Batch size tuned for your hardware
- [ ] Additional animal types added
- [ ] Custom training configurations tested
- [ ] Monitoring dashboard set up
- [ ] Backup strategy implemented
- [ ] Deployment pipeline configured

## 📝 Notes

Use this section to document your setup:

```
GPU Model: _______________
Max Batch Size: _______________
Typical Training Time (50 epochs): _______________
Dataset Size (total images): _______________
Best Configuration Found: _______________
```

## ✅ Final Sign-Off

- [ ] All items above checked
- [ ] System tested and working
- [ ] Ready to train DCGAN models
- [ ] Documentation reviewed
- [ ] Team members have access
- [ ] Backups configured

---

**You're ready to train! Start with the Quick Start Commands above. 🚀**

For issues, refer to:
- `GETTING_STARTED.md` - Basic setup guide
- `README.md` - Full documentation
- `GPU_SETUP.md` - GPU acceleration guide
- `DEPLOYMENT.md` - Cloud deployment
- `EXAMPLES.md` - Code examples
