# Deep Learning Projects Portfolio

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![React](https://img.shields.io/badge/React-18+-61dafb.svg)](https://react.dev/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Supported-2496ed.svg)](https://www.docker.com/)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)](#)

A comprehensive collection of production-ready deep learning projects showcasing expertise in neural networks, computer vision, and generative models. Each project includes full-stack implementations with training pipelines, REST APIs, and interactive frontends.

**[📖 Full Technical Details](./PROJECTS.md)** • **[🤝 Contributing Guide](./CONTRIBUTING.md)** • **[📄 License](./LICENSE)**

## 📊 Overview

This portfolio demonstrates advanced skills in:
- **Deep Learning Architecture Design** - CNNs, RNNs, GANs, Transformers
- **Full-Stack Development** - FastAPI backends, React frontends
- **Model Training & Optimization** - GPU acceleration, hyperparameter tuning
- **Production Deployment** - Docker, Railway, Vercel
- **Data Processing** - Complex preprocessing pipelines and visualization

---

## 🚀 Featured Projects

### 1. [Handwriting Recognition System](./Deep%20Neural%20Network) 
**Deep Neural Network for Character Recognition**

A complete end-to-end handwriting recognition system that segments handwritten text and recognizes individual characters using deep learning.

- **Tech Stack**: PyTorch, FastAPI, React, OpenCV
- **Models**: Character CNN, Specialist Models (digits/uppercase/lowercase)
- **Features**: 
  - Real-time handwriting recognition from canvas drawing
  - Multi-algorithm approach for optimal accuracy
  - Segmentation using connected component analysis
  - Interactive web interface with live results
  - Trained on EMNIST balanced dataset
- **Deployment**: Railway (backend) + Vercel (frontend)

[📖 Project Details](./Deep%20Neural%20Network/README.md) | [🔧 Setup Guide](./Deep%20Neural%20Network/SETUP_GUIDE.md)

---

### 2. [DCGAN Image Generation](./GAN)
**Deep Convolutional Generative Adversarial Network**

A sophisticated GAN system for generating high-quality animal images with real-time training monitoring and distributed GPU support.

- **Tech Stack**: PyTorch, FastAPI, React, PyYAML, Pillow
- **Features**:
  - Multi-animal support (cats, dogs, extensible)
  - Real-time training monitoring dashboard
  - Automatic GPU detection (NVIDIA CUDA, Apple MPS, CPU fallback)
  - YAML-based configuration system
  - Training visualization and sample generation
  - Modular architecture for easy customization
- **Deployment**: Docker containers, Railway, Vercel

[📖 Project Details](./GAN/README.md) | [🚀 Getting Started](./GAN/GETTING_STARTED.md) | [🔧 Deployment Guide](./GAN/DEPLOYMENT.md)

---

### 3. [RNN Text Generation System](./RNN)
**LSTM-based Sequential Text Generation**

A production-ready text generation system using recurrent neural networks with a comprehensive training pipeline and REST API.

- **Tech Stack**: TensorFlow/Keras, FastAPI, React, Pandas
- **Features**:
  - LSTM-based RNN architecture
  - Full training pipeline with data preprocessing
  - Configurable text generation with temperature control
  - Training history visualization
  - REST API for inference
  - Parameter control for output randomness
- **Deployment**: Docker, Railway

[📖 Project Details](./RNN/backend/README.md)

---

## 🛠️ Quick Start

Each project can be set up independently. Choose your project:

### Handwriting Recognition
```bash
cd "Deep Neural Network"
# See SETUP_GUIDE.md for detailed instructions
```

### DCGAN Training
```bash
cd GAN/Backend
pip install -r requirements.txt
python main.py
```

### RNN Text Generation
```bash
cd RNN/backend
pip install -r requirements.txt
python -m app.main
```

---

## 🏗️ Repository Structure

```
Deep-Learning-Projects/
├── README.md (this file)
├── CONTRIBUTING.md
├── LICENSE
├── package.json
│
├── Deep Neural Network/           # Character recognition system
│   ├── backend/                   # PyTorch models + FastAPI
│   ├── frontend/                  # React UI
│   ├── SETUP_GUIDE.md
│   └── README.md
│
├── GAN/                           # Image generation system
│   ├── Backend/                   # PyTorch DCGAN + FastAPI
│   ├── Frontend/                  # React dashboard
│   ├── GETTING_STARTED.md
│   ├── DEPLOYMENT.md
│   └── README.md
│
└── RNN/                           # Text generation system
    ├── backend/                   # TensorFlow LSTM + FastAPI
    ├── frontend/                  # React UI
    └── README.md
```

---

## 🔑 Key Technologies

**Machine Learning & Deep Learning:**
- PyTorch - Deep neural networks and GANs
- TensorFlow/Keras - LSTM and sequential models
- NumPy, Pandas - Data processing and analysis
- OpenCV - Computer vision operations

**Backend:**
- FastAPI - High-performance REST APIs
- Python 3.9+ - Core implementation language
- Docker - Containerization and deployment

**Frontend:**
- React - Interactive user interfaces
- Vite - Modern frontend tooling
- CSS3 - Styling and responsive design

**Deployment:**
- Railway - Backend hosting
- Vercel - Frontend hosting
- Docker - Container orchestration

---

## 📋 Features Across All Projects

✅ **Production Ready** - Error handling, validation, logging  
✅ **Full Documentation** - Setup guides, API docs, code comments  
✅ **GPU Optimized** - Automatic device detection and acceleration  
✅ **RESTful APIs** - Clean, documented endpoints  
✅ **Interactive Frontends** - Real-time feedback and visualization  
✅ **Modular Design** - Reusable components and architecture  
✅ **Cloud Ready** - Docker support and deployment configs  
✅ **Version Controlled** - Git history with meaningful commits  

---

## 🚀 Deployment Status

- **Handwriting Recognition**: [Live on Railway](https://handwriting-api.railway.app) + [Frontend on Vercel](https://handwriting-recognition-ui.vercel.app)
- **DCGAN System**: Ready for Railway/Vercel deployment
- **RNN Text Generation**: Ready for Railway/Vercel deployment

---

## 📈 Skills Demonstrated

### Machine Learning
- Neural Network Architecture Design
- Model Training & Optimization
- Hyperparameter Tuning
- Data Augmentation & Preprocessing
- GPU Acceleration (CUDA, MPS)

### Software Engineering
- Full-Stack Development
- REST API Design
- React Component Architecture
- Docker Containerization
- CI/CD & Deployment
- Code Organization & Best Practices

### DevOps & Cloud
- Docker & Container Management
- Cloud Deployment (Railway, Vercel)
- Environment Configuration
- Scaling & Performance Optimization

---

## 📖 Documentation

- **[CONTRIBUTING.md](./CONTRIBUTING.md)** - Guidelines for contributions
- **[LICENSE](./LICENSE)** - MIT License
- Each project has its own detailed README and setup guides

---

## 🤝 Contributing

This is a portfolio project, but contributions and suggestions are welcome! See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

---

## 📄 License

All projects in this repository are licensed under the MIT License. See [LICENSE](./LICENSE) for details.

---

## 📞 Contact & Links

- **GitHub**: [Your GitHub Profile](https://github.com/yourusername)
- **Portfolio**: [Your Portfolio Website]
- **LinkedIn**: [Your LinkedIn Profile]

---

**Last Updated**: December 2024  
**Status**: Active Development & Maintenance
