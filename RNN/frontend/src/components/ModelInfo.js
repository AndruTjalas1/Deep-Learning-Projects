import React, { useState, useEffect } from 'react';
import { rnnApi as api } from '../services/rnnApi';
import './ModelInfo.css';

const ModelInfo = () => {
  const [modelInfo, setModelInfo] = useState(null);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchModelInfo();
  }, []);

  const fetchModelInfo = async () => {
    try {
      setLoading(true);
      setError('');
      
      const infoResponse = await api.modelInfo();
      setModelInfo(infoResponse);
      
      try {
        const statsResponse = await api.modelInfo();
        setStats(statsResponse);
      } catch (statsError) {
        console.log('Could not fetch stats:', statsError);
      }
    } catch (err) {
      setError(`Failed to load model info: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <div className="model-info"><p>Loading model information...</p></div>;
  }

  if (error) {
    return (
      <div className="model-info">
        <div className="error-message">{error}</div>
        <button onClick={fetchModelInfo}>Retry</button>
      </div>
    );
  }

  return (
    <div className="model-info">
      <h2>📊 Model Information</h2>
      
      {modelInfo && (
        <div className="info-grid">
          <div className="info-card">
            <h3>Vocabulary Size</h3>
            <p className="info-value">{modelInfo.vocabulary_size?.toLocaleString()}</p>
            <small>unique words</small>
          </div>

          <div className="info-card">
            <h3>Sequence Length</h3>
            <p className="info-value">{modelInfo.sequence_length}</p>
            <small>tokens per sequence</small>
          </div>

          <div className="info-card">
            <h3>Embedding Dimension</h3>
            <p className="info-value">{modelInfo.embedding_dim}</p>
            <small>vector dimensions</small>
          </div>

          <div className="info-card">
            <h3>LSTM Units</h3>
            <p className="info-value">{modelInfo.lstm_units}</p>
            <small>per layer</small>
          </div>

          <div className="info-card">
            <h3>LSTM Layers</h3>
            <p className="info-value">{modelInfo.num_layers}</p>
            <small>stacked layers</small>
          </div>

          <div className="info-card">
            <h3>Status</h3>
            <p className="info-value" style={{ color: modelInfo.is_loaded ? '#4CAF50' : '#ff9800' }}>
              {modelInfo.is_loaded ? '✅ Loaded' : '⚠️ Not Ready'}
            </p>
            <small>model status</small>
          </div>
        </div>
      )}

      {stats && stats.training_stats && (
        <div className="training-stats">
          <h3>📈 Training Statistics</h3>
          <div className="stats-grid">
            <div className="stat-item">
              <span>Final Loss:</span>
              <strong>{stats.training_stats.final_loss?.toFixed(4)}</strong>
            </div>
            <div className="stat-item">
              <span>Final Accuracy:</span>
              <strong>{(stats.training_stats.final_accuracy * 100)?.toFixed(2)}%</strong>
            </div>
            <div className="stat-item">
              <span>Val Loss:</span>
              <strong>{stats.training_stats.final_val_loss?.toFixed(4)}</strong>
            </div>
            <div className="stat-item">
              <span>Val Accuracy:</span>
              <strong>{(stats.training_stats.final_val_accuracy * 100)?.toFixed(2)}%</strong>
            </div>
            <div className="stat-item">
              <span>Epochs Trained:</span>
              <strong>{stats.training_stats.epochs_trained}</strong>
            </div>
            <div className="stat-item">
              <span>Total Parameters:</span>
              <strong>{(stats.model_parameters / 1000000)?.toFixed(2)}M</strong>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelInfo;
