import React, { useState, useEffect } from 'react';
import trainingApi from '../api';
import './ConfigPanel.css';

export default function ConfigPanel({ onConfigUpdate, isTraining }) {
  const [config, setConfig] = useState(null);
  const [formData, setFormData] = useState({
    epochs: 50,
    batch_size: 64,
    learning_rate: 0.0002,
    resolution: 64,
    sample_interval: 100,
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchConfig();
  }, []);

  const fetchConfig = async () => {
    try {
      setLoading(true);
      const data = await trainingApi.getConfig();
      setConfig(data);
      setFormData({
        epochs: data.training.epochs,
        batch_size: data.training.batch_size,
        learning_rate: data.training.learning_rate,
        resolution: data.image.resolution,
        sample_interval: data.sampling.sample_interval,
      });
      setError(null);
    } catch (err) {
      setError(`Failed to load config: ${err.message}`);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    const parsedValue = ['learning_rate'].includes(name) ? parseFloat(value) : parseInt(value);
    setFormData(prev => ({
      ...prev,
      [name]: parsedValue,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const result = await trainingApi.updateConfig({
        epochs: formData.epochs,
        batch_size: formData.batch_size,
        learning_rate: formData.learning_rate,
        resolution: formData.resolution,
        sample_interval: formData.sample_interval,
      });
      setConfig(result.config);
      onConfigUpdate?.(result.config);
      setError(null);
      alert('Configuration updated successfully!');
    } catch (err) {
      setError(`Failed to update config: ${err.message}`);
      console.error(err);
    }
  };

  if (loading) {
    return <div className="config-panel"><p>Loading configuration...</p></div>;
  }

  return (
    <div className="config-panel">
      <h2>⚙️ Configuration</h2>

      {error && <div className="error-message">{error}</div>}

      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="epochs">Epochs</label>
          <input
            id="epochs"
            type="number"
            name="epochs"
            value={formData.epochs}
            onChange={handleInputChange}
            disabled={isTraining}
            min="1"
            max="1000"
          />
          <small>Number of training epochs</small>
        </div>

        <div className="form-group">
          <label htmlFor="batch_size">Batch Size</label>
          <input
            id="batch_size"
            type="number"
            name="batch_size"
            value={formData.batch_size}
            onChange={handleInputChange}
            disabled={isTraining}
            min="8"
            max="512"
            step="8"
          />
          <small>Images per batch</small>
        </div>

        <div className="form-group">
          <label htmlFor="learning_rate">Learning Rate</label>
          <input
            id="learning_rate"
            type="number"
            name="learning_rate"
            value={formData.learning_rate}
            onChange={handleInputChange}
            disabled={isTraining}
            min="0.00001"
            max="0.1"
            step="0.00001"
          />
          <small>Optimizer learning rate</small>
        </div>

        <div className="form-group">
          <label htmlFor="resolution">Image Resolution</label>
          <select
            id="resolution"
            name="resolution"
            value={formData.resolution}
            onChange={handleInputChange}
            disabled={isTraining}
          >
            <option value="64">64x64</option>
            <option value="128">128x128</option>
            <option value="256">256x256</option>
          </select>
          <small>Output image size</small>
        </div>

        <div className="form-group">
          <label htmlFor="sample_interval">Sample Interval</label>
          <input
            id="sample_interval"
            type="number"
            name="sample_interval"
            value={formData.sample_interval}
            onChange={handleInputChange}
            disabled={isTraining}
            min="10"
            max="1000"
            step="10"
          />
          <small>Generate samples every N batches</small>
        </div>

        <button type="submit" disabled={isTraining} className="submit-btn">
          {isTraining ? 'Training in progress...' : 'Update Configuration'}
        </button>
      </form>
    </div>
  );
}
