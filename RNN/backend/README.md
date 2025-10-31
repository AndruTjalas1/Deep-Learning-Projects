# RNN Text Generation System

A complete machine learning project demonstrating how to build, train, and deploy a **Recurrent Neural Network (RNN)** for intelligent text generation using deep learning.

## 📋 Project Overview

This project implements a **text generation system** using LSTM (Long Short-Term Memory) networks. It includes:

- **Backend**: FastAPI REST API server for serving the trained model
- **Core ML**: TensorFlow/Keras implementation of LSTM text generation
- **Training Pipeline**: Complete training orchestration with visualization
- **Data Processing**: Text preprocessing, tokenization, and sequence generation
- **Deployment Ready**: Designed for cloud deployment

### Key Features

✅ **LSTM-based RNN** - Deep learning model for sequential text processing  
✅ **Full Training Pipeline** - Data preprocessing → Model building → Training → Evaluation  
✅ **REST API** - FastAPI backend for generating text via HTTP endpoints  
✅ **Interactive Frontend** - React UI for real-time text generation  
✅ **Visualizations** - Training history plots and model architecture diagrams  
✅ **Parameter Control** - Adjustable temperature for controlling output randomness  
✅ **Production Ready** - Error handling, validation, CORS support  

---

## 🏗️ Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── text_generator.py      # Core ML logic (TextGenerator class)
│   ├── train.py               # Training script
│   ├── main.py                # FastAPI server
│   └── models.py              # Pydantic request/response models
├── data/
│   └── training_text.txt      # Training corpus (download required)
├── saved_models/
│   ├── model.h5               # Trained LSTM weights
│   ├── tokenizer.pkl          # Word vocabulary
│   └── config.json            # Model configuration
├── visualizations/
│   ├── model_architecture.png
│   └── training_history.png
├── scripts/
│   └── download_data.py       # Download training data from Project Gutenberg
└── requirements.txt           # Python dependencies

frontend/                       # React UI (optional)
├── src/
├── public/
└── package.json
```

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.8+
- pip (Python package manager)
- CUDA/GPU (optional but recommended for faster training)

### 2. Clone and Setup

```bash
# Navigate to project directory
cd backend

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Download Training Data

The model needs text data to learn from. Download from Project Gutenberg:

```bash
# Navigate to scripts directory
cd scripts

# Run download script
python download_data.py

# Follow prompts to select and download a book
# (Recommended: Alice in Wonderland or Pride and Prejudice)
```

Available books:
- Alice's Adventures in Wonderland (~170 KB)
- Pride and Prejudice (~650 KB)
- The Adventures of Sherlock Holmes (~600 KB)
- Frankenstein (~445 KB)
- The Great Gatsby (~300 KB)

### 4. Train the Model

```bash
# Navigate back to app directory
cd ../app

# Run training pipeline
python train.py
```

This will:
1. Load and preprocess training data
2. Build LSTM model architecture
3. Train for 100 epochs
4. Generate visualizations
5. Save model and tokenizer to `saved_models/`

Expected training time: 5-30 minutes (depending on hardware)

### 5. Start the API Server

```bash
# From app directory
python main.py
```

Server will start at: `http://localhost:8000`

### 6. Test the API

Visit the interactive API documentation:

```
http://localhost:8000/docs
```

Or generate text via curl:

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "seed_text": "the cat sat on",
    "num_words": 20,
    "temperature": 1.0
  }'
```

---

## 📚 API Endpoints

### Health Check
```
GET /
GET /health
```
Check if the model is loaded and server is running.

### Model Information
```
GET /model/info
GET /model/summary
GET /stats
```
Get model architecture, configuration, and training statistics.

### Text Generation
```
POST /generate
```

**Request:**
```json
{
  "seed_text": "the cat sat on",
  "num_words": 20,
  "temperature": 1.0
}
```

**Response:**
```json
{
  "seed_text": "the cat sat on",
  "generated_text": "the cat sat on the mat and slept peacefully...",
  "num_words_generated": 20,
  "temperature": 1.0
}
```

### Visualizations
```
GET /visualizations/architecture
GET /visualizations/training
```
Download model architecture diagram and training history plots.

---

## 🧠 Understanding the Architecture

### LSTM Layer Equations

The model uses LSTM cells to process sequences:

**Forget Gate** (what to remove):
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

**Input Gate** (what to add):
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

**Candidate Cell State**:
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

**Cell State Update**:
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

**Output Gate**:
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

**Hidden State**:
$$h_t = o_t \odot \tanh(C_t)$$

### Model Architecture

```
Input (sequence of word IDs)
    ↓
Embedding Layer (100 dims)
    ↓
LSTM Layer 1 (150 units, return_sequences=True)
    ↓
Dropout (0.2)
    ↓
LSTM Layer 2 (150 units)
    ↓
Dropout (0.2)
    ↓
Dense Output Layer (vocabulary_size units, softmax)
    ↓
Output (probability distribution over vocabulary)
```

### Text Generation Process

1. **Seed Text**: "the cat sat on"
2. **Tokenization**: Convert words to IDs [1, 2, 3, 4]
3. **Padding**: Pad to sequence length 50
4. **LSTM Processing**: Feed through network
5. **Prediction**: Get probability distribution for next word
6. **Temperature Scaling**: Adjust randomness
7. **Sampling**: Randomly select next word based on probabilities
8. **Repeat**: Add predicted word and repeat for desired number of words

---

## 🎛️ Hyperparameters

Edit these in `train.py`:

```python
SEQUENCE_LENGTH = 50          # Length of input sequences
EMBEDDING_DIM = 100           # Embedding vector dimension
LSTM_UNITS = 150              # Units in LSTM layers
NUM_LSTM_LAYERS = 2           # Number of stacked LSTM layers
DROPOUT_RATE = 0.2            # Dropout for regularization
EPOCHS = 100                  # Training epochs
BATCH_SIZE = 128              # Batch size for training
```

### Temperature (Generation)

Control the "creativity" of generated text:

- **Temperature < 1.0**: More deterministic, repeats likely words
- **Temperature = 1.0**: Balanced (default)
- **Temperature > 1.0**: More diverse, more random

Example:
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "seed_text": "once upon a time",
    "num_words": 30,
    "temperature": 0.5
  }'
```

---

## 📊 Training Process

### Data Pipeline

```
Raw Text File
    ↓
Text Preprocessing (lowercase, remove special chars)
    ↓
Tokenization (build vocabulary)
    ↓
Sequence Generation (sliding window)
    ↓
Padding (uniform sequence length)
    ↓
One-hot Encoding (prepare for training)
    ↓
Training Data Ready
```

### Training Loop

```python
for epoch in range(100):
    for batch in training_data:
        # Forward pass
        predictions = model(batch_X)
        
        # Calculate loss
        loss = categorical_crossentropy(batch_y, predictions)
        
        # Backward pass
        gradients = compute_gradients(loss)
        
        # Update weights
        optimizer.apply_gradients(gradients)
```

---

## 🐛 Troubleshooting

### "Model not loaded" Error

**Solution**: Make sure you've trained the model first:
```bash
python train.py
```

### "Training data not found"

**Solution**: Download training data:
```bash
cd scripts
python download_data.py
```

### Out of Memory (OOM)

**Solutions**:
1. Reduce `BATCH_SIZE` in `train.py` (e.g., 64 instead of 128)
2. Reduce `SEQUENCE_LENGTH` (e.g., 30 instead of 50)
3. Reduce `LSTM_UNITS` (e.g., 100 instead of 150)
4. Use a smaller training dataset

### Slow Training

**Solutions**:
1. Use GPU if available (CUDA must be installed)
2. Reduce model complexity (fewer layers, fewer units)
3. Use smaller batch size for GPU memory efficiency

---

## 🔧 Configuration

Model configuration is saved in `saved_models/config.json`:

```json
{
    "sequence_length": 50,
    "embedding_dim": 100,
    "lstm_units": 150,
    "num_lstm_layers": 2,
    "dropout_rate": 0.2,
    "vocab_size": 8532
}
```

---

## 📈 Monitoring Training

During training, you'll see:

```
Epoch 1/100
243/243 [==============================] - 45s 186ms/step - loss: 5.1234 - accuracy: 0.0892 - 
        val_loss: 4.8456 - val_accuracy: 0.1023
Epoch 2/100
243/243 [==============================] - 42s 173ms/step - loss: 4.7123 - accuracy: 0.1234 -
        val_loss: 4.5678 - val_accuracy: 0.1456
...
```

Key metrics:
- **loss**: Training error (should decrease)
- **accuracy**: Training accuracy (should increase)
- **val_loss**: Validation error (should decrease)
- **val_accuracy**: Validation accuracy (should increase)

---

## 💾 Model Files

After training, you'll have:

1. **model.h5** (~50-100 MB): LSTM weights and architecture
2. **tokenizer.pkl** (~1-2 MB): Word vocabulary mapping
3. **config.json** (< 1 KB): Model configuration

---

## 🚢 Deployment

### Local Deployment

Already covered in "Quick Start" above.

### Cloud Deployment (Render, Railway, Heroku)

1. Create a `Procfile`:
```
web: cd backend && python app/main.py
```

2. Create a `.gitignore`:
```
venv/
*.pyc
__pycache__/
.DS_Store
saved_models/
visualizations/
```

3. Push to GitHub and connect to Render/Railway

### Docker Deployment

Create `Dockerfile`:
```dockerfile
FROM python:3.9

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install -r requirements.txt

COPY backend ./

CMD ["python", "app/main.py"]
```

---

## 🎓 Learning Outcomes

After completing this project, you'll understand:

1. **RNN Fundamentals**: How recurrent networks process sequences
2. **LSTM Architecture**: Gating mechanisms, cell state, hidden state
3. **Text Preprocessing**: Tokenization, embedding, sequence creation
4. **Deep Learning Training**: Backpropagation through time, optimization
5. **Model Evaluation**: Loss, accuracy, overfitting, regularization
6. **API Development**: REST endpoints, request validation, error handling
7. **Full-Stack ML**: From data to deployed model serving

---

## 📚 Further Learning

### RNN Concepts
- [Colah's LSTM Blog Post](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Karpathy's RNN Effectiveness](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [Stanford NLP with RNNs](https://web.stanford.edu/class/cs224n/)

### Implementation References
- [TensorFlow Text Generation](https://www.tensorflow.org/tutorials/sequences/text_generation)
- [Keras LSTM Documentation](https://keras.io/api/layers/recurrent_layers/lstm/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

### Datasets
- [Project Gutenberg](https://www.gutenberg.org/)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [Common Crawl](https://commoncrawl.org/)

---

## 📝 License

This project is for educational purposes. You are free to use, modify, and distribute it.

---

## 💡 Tips for Best Results

1. **Larger Dataset**: More training data = better results. Try combining multiple books.
2. **Longer Training**: Train for more epochs (but watch for overfitting).
3. **Experiment with Temperature**: Lower values for coherent text, higher for creativity.
4. **Pre-trained Embeddings**: Consider using GloVe or Word2Vec embeddings.
5. **Data Quality**: Clean, well-formatted text produces better results.

---

## 🤝 Contributing

Feel free to extend this project with:
- Attention mechanisms
- Beam search for generation
- Different RNN architectures (GRU, Transformers)
- Web-based UI
- Multi-model support
- Language-specific preprocessing

---

**Built with ❤️ for learning deep learning**
