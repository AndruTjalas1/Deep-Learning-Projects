"""
TextGenerator: RNN-based text generation with LSTM architecture using PyTorch
Handles text preprocessing, model building, training, and generation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import re
import json
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
try:
    import seaborn as sns
except ImportError:
    sns = None
import os
from pathlib import Path

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {DEVICE}")


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_units, num_lstm_layers, dropout_rate,
                 activation_fn='relu', embedding_weights=None, trainable_embeddings=True,
                 recurrent_dropout=0.0):
        super().__init__()
        if embedding_weights is not None:
            self.embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(embedding_weights),
                freeze=not trainable_embeddings,
                padding_idx=0
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.embedding_dropout = nn.Dropout(dropout_rate * 0.3)
        self.lstm = nn.LSTM(
            embedding_dim, lstm_units, num_lstm_layers,
            dropout=(dropout_rate if num_lstm_layers > 1 else 0.0),
            batch_first=True, bidirectional=False
        )
        self.dense = nn.Linear(lstm_units, 512)
        self.layer_norm = nn.LayerNorm(512)
        self.dropout = nn.Dropout(dropout_rate)
        self.output = nn.Linear(512, vocab_size)

        if activation_fn == 'gelu':
            self.activation = nn.GELU()
        elif activation_fn == 'elu':
            self.activation = nn.ELU()
        elif activation_fn == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x)
        x = self.embedding_dropout(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dense(x)
        x = self.layer_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.output(x)
        return x


class TextGenerator:
    def __init__(self, 
                 sequence_length: int = 50,
                 embedding_dim: int = 100,
                 lstm_units: int = 150,
                 num_lstm_layers: int = 2,
                 dropout_rate: float = 0.2,
                 recurrent_dropout: float = 0.0,
                 vocab_size: Optional[int] = None,
                 activation_fn: str = 'relu',
                 use_glove_embeddings: bool = False,
                 trainable_embeddings: bool = True):
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.num_lstm_layers = num_lstm_layers
        self.dropout_rate = dropout_rate
        self.recurrent_dropout = recurrent_dropout
        self.activation_fn = activation_fn
        self.use_glove_embeddings = use_glove_embeddings
        self.trainable_embeddings = trainable_embeddings
        self.vocab_size = vocab_size

        self.tokenizer = None
        self.index_to_word = None
        self.model: Optional[nn.Module] = None
        self.history = None
        self.config = None
        self.device = DEVICE

    # ---------- preprocessing ----------
    def preprocess_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'\*\*\*.*?(START|END).*?\*\*\*', '', text, flags=re.DOTALL)
        text = re.sub(r"[^a-z0-9\s\.\,\!\?\'\-\"\:\;\(\)—]", '', text)
        text = ' '.join(text.split())
        words = [w for w in text.split() if w.strip()]
        return ' '.join(words)

    def prepare_sequences(self, text: str, min_word_freq: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        words = text.split()
        from collections import Counter
        word_counts = Counter(words)
        frequent = {w for w, c in word_counts.items() if c >= min_word_freq}
        filtered = [w if w in frequent else '<OOV>' for w in words]
        vocab = {'<OOV>': 0}
        vocab.update({w: i + 1 for i, w in enumerate(sorted({w for w in filtered if w != '<OOV>'}))})

        class SimpleTokenizer:
            def __init__(self, word_index): self.word_index = word_index
            def texts_to_sequences(self, texts):
                out = []
                for t in texts:
                    out.append([self.word_index.get(w, 0) for w in t.split()])
                return out

        self.tokenizer = SimpleTokenizer(vocab)
        self.index_to_word = {idx: w for w, idx in vocab.items()}

        seq = [vocab.get(w, 0) for w in filtered]
        X, y = [], []
        for i in range(len(seq) - self.sequence_length):
            X.append(seq[i:i + self.sequence_length])
            y.append(seq[i + self.sequence_length])
        return np.array(X, dtype=np.int64), np.array(y, dtype=np.int64)

    # ---------- build/train ----------
    def build_model(self, vocab_size: int):
        self.model = LSTMModel(
            vocab_size, self.embedding_dim, self.lstm_units, self.num_lstm_layers,
            self.dropout_rate, self.activation_fn
        ).to(self.device)
        self.vocab_size = vocab_size
        return self.model

    # ---------- save/load ----------
    def save_model(self, model_dir: str = 'saved_models') -> None:
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(model_dir, 'model.pt'))
        if self.tokenizer is not None:
            with open(os.path.join(model_dir, 'tokenizer.json'), 'w') as f:
                json.dump(self.tokenizer.word_index, f, indent=2)
        cfg = {
            'sequence_length': self.sequence_length,
            'embedding_dim': self.embedding_dim,
            'lstm_units': self.lstm_units,
            'num_lstm_layers': self.num_lstm_layers,
            'dropout_rate': self.dropout_rate,
            'vocab_size': len(self.tokenizer.word_index) if self.tokenizer else self.vocab_size or 0,
        }
        with open(os.path.join(model_dir, 'config.json'), 'w') as f:
            json.dump(cfg, f, indent=2)

    def load_model(self, model_dir: str = 'saved_models') -> None:
        # required
        with open(os.path.join(model_dir, 'config.json'), 'r') as f:
            self.config = json.load(f)

        self.sequence_length = self.config.get('sequence_length', 30)
        self.embedding_dim = self.config.get('embedding_dim', 50)
        self.lstm_units = self.config.get('lstm_units', 75)
        self.num_lstm_layers = self.config.get('num_lstm_layers', 1)
        self.dropout_rate = self.config.get('dropout_rate', 0.2)
        vocab_size = self.config.get('vocab_size', 2000)

        # tokenizer: prefer JSON (avoids _pickle.UnpicklingError)
        tok_json = os.path.join(model_dir, 'tokenizer.json')
        tok_pkl  = os.path.join(model_dir, 'tokenizer.pkl')

        if os.path.exists(tok_json):
            with open(tok_json, 'r') as f:
                word_index = json.load(f)

            class SimpleTokenizer:
                def __init__(self, word_index): self.word_index = word_index
                def texts_to_sequences(self, texts):
                    out = []
                    for t in texts:
                        out.append([self.word_index.get(w, 0) for w in t.split()])
                    return out

            self.tokenizer = SimpleTokenizer(word_index)
            self.index_to_word = {int(idx): w for w, idx in word_index.items()}
            print(f"✓ Tokenizer loaded from JSON ({tok_json})")
        elif os.path.exists(tok_pkl):
            # legacy fallback
            with open(tok_pkl, 'rb') as f:
                self.tokenizer = pickle.load(f)
            # build reverse index safely if missing
            wi = getattr(self.tokenizer, "word_index", {}) or {}
            self.index_to_word = {int(idx): w for w, idx in wi.items()}
            print(f"✓ Tokenizer loaded from PKL ({tok_pkl})")
        else:
            print("⚠ No tokenizer found; generation will be limited until rebuilt.")
            self.tokenizer = None
            self.index_to_word = {}

        # build and load torch weights
        self.build_model(vocab_size)
        model_pt = os.path.join(model_dir, 'model.pt')
        if os.path.exists(model_pt):
            self.model.load_state_dict(torch.load(model_pt, map_location=self.device))
            self.model.eval()
            print(f"✓ Torch model loaded from {model_pt}")
        else:
            # optional: legacy Keras .h5 could be handled elsewhere if needed
            raise FileNotFoundError(f"Model state dict not found at {model_pt}")

    # ---------- generation ----------
    def _safe_sample(self, probs: np.ndarray) -> int:
        probs = np.nan_to_num(np.asarray(probs), nan=0.0)
        total = probs.sum()
        if total <= 0 or np.isclose(total, 0.0):
            return int(np.argmax(probs))
        probs = probs / total
        try:
            return int(np.random.choice(len(probs), p=probs))
        except Exception:
            return int(np.argmax(probs))

    def generate_text(self,
                      seed_text: str,
                      num_words: int = 50,
                      temperature: float = 0.8,
                      top_k: int = 40,
                      top_p: float = 0.85,
                      use_beam_search: bool = True,
                      beam_width: int = 3) -> str:
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Please ensure tokenizer.json or .pkl exists.")

        seed = self.preprocess_text(seed_text)
        if use_beam_search:
            return self._generate_beam_search(seed, num_words, beam_width, temperature)

        generated = seed
        oov_idx = int(self.tokenizer.word_index.get('<OOV>', 0))

        with torch.no_grad():
            for _ in range(num_words):
                token_list = self.tokenizer.texts_to_sequences([generated])[0]
                token_list = token_list[-self.sequence_length:]
                if len(token_list) < self.sequence_length:
                    token_list = [0] * (self.sequence_length - len(token_list)) + token_list

                input_t = torch.tensor([token_list], dtype=torch.long).to(self.device)
                logits = self.model(input_t)[0].cpu().numpy()
                logits = logits / max(temperature, 0.01)

                # forbid OOV token
                if 0 <= oov_idx < logits.shape[0]:
                    logits[oov_idx] = -np.inf

                # top-k
                if top_k > 0 and top_k < len(logits):
                    bad = np.argsort(logits)[:-top_k]
                    logits[bad] = -np.inf

                # top-p
                if 0 < top_p < 1.0:
                    order = np.argsort(logits)[::-1]
                    sorted_logits = logits[order]
                    exp_logits = np.exp(sorted_logits - np.max(sorted_logits))
                    cumsum = np.cumsum(exp_logits / np.sum(exp_logits))
                    to_remove = cumsum > top_p
                    if np.any(to_remove):
                        to_remove[0] = False
                        logits[order[to_remove]] = -np.inf

                exp = np.exp(logits - np.max(logits))
                probs = exp / (np.sum(exp) + 1e-10)

                idx = self._safe_sample(probs)
                word = self.index_to_word.get(int(idx))
                if not word or word == '<OOV>':
                    # fallback to argmax non-OOV
                    if 0 <= oov_idx < probs.shape[0]:
                        probs[oov_idx] = -np.inf
                    idx = int(np.argmax(probs))
                    word = self.index_to_word.get(idx, None)

                if word and word != '<OOV>':
                    generated += ' ' + word

        return generated

    def _generate_beam_search(self, seed_text: str, num_words: int, beam_width: int = 3,
                              temperature: float = 0.8) -> str:
        reverse_idx = self.index_to_word or {}
        beams = [(seed_text, 0.0)]
        oov_idx = int(self.tokenizer.word_index.get('<OOV>', 0))

        with torch.no_grad():
            for _ in range(num_words):
                new_beams = []
                for text, logp in beams:
                    token_list = self.tokenizer.texts_to_sequences([text])[0]
                    token_list = token_list[-self.sequence_length:]
                    if len(token_list) < self.sequence_length:
                        token_list = [0] * (self.sequence_length - len(token_list)) + token_list

                    input_t = torch.tensor([token_list], dtype=torch.long).to(self.device)
                    logits = self.model(input_t)[0].cpu().numpy()
                    logits = logits / max(temperature, 0.01)

                    if 0 <= oov_idx < logits.shape[0]:
                        logits[oov_idx] = -np.inf

                    exp = np.exp(logits - np.max(logits))
                    probs = exp / (np.sum(exp) + 1e-10)

                    top_idx = np.argsort(probs)[-beam_width:]
                    for idx in reversed(top_idx):
                        w = reverse_idx.get(int(idx))
                        if w and w != '<OOV>':
                            new_beams.append((text + ' ' + w, logp + float(np.log(max(probs[idx], 1e-12)))))

                new_beams.sort(key=lambda x: x[1], reverse=True)
                beams = new_beams[:beam_width] if new_beams else beams

        return beams[0][0] if beams else seed_text

    # optional helpers
    def get_model_summary(self) -> str:
        return str(self.model)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
