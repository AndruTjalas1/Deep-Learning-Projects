import React, { useState, useEffect } from 'react';
import trainingApi from '../api';
import './TrainingControl.css';

export default function TrainingControl({ onTrainingStart, onTrainingStop, isTraining }) {
  const [status, setStatus] = useState(null);
  const [animalTypes, setAnimalTypes] = useState(['cats', 'dogs']);
  const [epochs, setEpochs] = useState(50);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const pollStatus = async () => {
      try {
        const data = await trainingApi.getTrainingStatus();
        setStatus(data);
      } catch (err) {
        console.error('Error polling status:', err);
      }
    };

    const interval = setInterval(pollStatus, 2000);
    return () => clearInterval(interval);
  }, []);

  const handleStartTraining = async () => {
    try {
      setLoading(true);
      setError(null);
      const result = await trainingApi.startTraining(animalTypes, epochs);
      setStatus({ training_active: true });
      onTrainingStart?.(result);
    } catch (err) {
      setError(`Failed to start training: ${err.message}`);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleStopTraining = async () => {
    try {
      setLoading(true);
      setError(null);
      await trainingApi.stopTraining();
      onTrainingStop?.();
    } catch (err) {
      setError(`Failed to stop training: ${err.message}`);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const toggleAnimalType = (type) => {
    setAnimalTypes(prev =>
      prev.includes(type) ? prev.filter(t => t !== type) : [...prev, type]
    );
  };

  const trainingProgress = status?.current_epoch
    ? Math.round((status.current_epoch / status.total_epochs) * 100)
    : 0;

  return (
    <div className="training-control">
      <h2>🚀 Training Control</h2>

      {error && <div className="error-message">{error}</div>}

      <div className="control-section">
        <h3>Animal Types</h3>
        <div className="animal-selector">
          {['cats', 'dogs'].map(type => (
            <label key={type} className="checkbox-label">
              <input
                type="checkbox"
                checked={animalTypes.includes(type)}
                onChange={() => toggleAnimalType(type)}
                disabled={isTraining || status?.training_active}
              />
              <span>{type.charAt(0).toUpperCase() + type.slice(1)}</span>
            </label>
          ))}
        </div>
      </div>

      <div className="control-section">
        <label htmlFor="epochs-input">Training Epochs: {epochs}</label>
        <input
          id="epochs-input"
          type="range"
          min="1"
          max="200"
          value={epochs}
          onChange={(e) => setEpochs(parseInt(e.target.value))}
          disabled={isTraining || status?.training_active}
          className="slider"
        />
      </div>

      {status?.training_active && (
        <div className="training-progress">
          <h4>Training Progress</h4>
          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{ width: `${trainingProgress}%` }}
            />
          </div>
          <div className="progress-text">
            Epoch: {status.current_epoch + 1} / {status.total_epochs}
            ({trainingProgress}%)
          </div>
          {status.latest_g_loss && (
            <div className="loss-display">
              <div>G Loss: {status.latest_g_loss.toFixed(4)}</div>
              <div>D Loss: {status.latest_d_loss.toFixed(4)}</div>
            </div>
          )}
        </div>
      )}

      <div className="button-group">
        <button
          onClick={handleStartTraining}
          disabled={loading || status?.training_active}
          className="btn btn-primary"
        >
          {loading ? 'Starting...' : 'Start Training'}
        </button>
        <button
          onClick={handleStopTraining}
          disabled={loading || !status?.training_active}
          className="btn btn-danger"
        >
          {loading ? 'Stopping...' : 'Stop Training'}
        </button>
      </div>
    </div>
  );
}
