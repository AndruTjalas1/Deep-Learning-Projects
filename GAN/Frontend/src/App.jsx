import React, { useState, useEffect } from 'react';
import ConfigPanel from './components/ConfigPanel';
import TrainingControl from './components/TrainingControl';
import MetricsDisplay from './components/MetricsDisplay';
import GalleryView from './components/GalleryView';
import trainingApi from './api';
import './App.css';

function App() {
  const [isTraining, setIsTraining] = useState(false);
  const [deviceInfo, setDeviceInfo] = useState(null);
  const [refreshGallery, setRefreshGallery] = useState(0);

  useEffect(() => {
    // Fetch device info on mount
    const fetchDeviceInfo = async () => {
      try {
        const info = await trainingApi.getDeviceInfo();
        setDeviceInfo(info);
      } catch (err) {
        console.error('Error fetching device info:', err);
      }
    };

    fetchDeviceInfo();

    // Check training status periodically
    const statusInterval = setInterval(async () => {
      try {
        const status = await trainingApi.getTrainingStatus();
        setIsTraining(status.training_active || false);
      } catch (err) {
        console.error('Error checking training status:', err);
      }
    }, 3000);

    return () => clearInterval(statusInterval);
  }, []);

  const handleTrainingStart = () => {
    setIsTraining(true);
  };

  const handleTrainingStop = () => {
    setIsTraining(false);
  };

  const handleConfigUpdate = () => {
    // Refresh gallery when config changes
    setRefreshGallery(prev => prev + 1);
  };

  const getDeviceString = () => {
    if (!deviceInfo) return 'Loading...';
    if (deviceInfo.cuda_available) {
      return `NVIDIA GPU: ${deviceInfo.cuda_device_name}`;
    }
    if (deviceInfo.mps_available) {
      return 'Apple Metal Performance Shaders (MPS)';
    }
    return 'CPU';
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>🐱🐶 DCGAN Generator</h1>
        <p>Deep Convolutional Generative Adversarial Network for Cats & Dogs</p>
        <div className="device-status">
          <span>Device: {getDeviceString()}</span>
          <span className={`status-badge ${isTraining ? 'training' : 'idle'}`}>
            {isTraining ? '🔴 Training' : '🟢 Idle'}
          </span>
        </div>
      </header>

      <main className="app-main">
        <div className="container">
          <div className="grid-layout">
            <div className="column">
              <ConfigPanel isTraining={isTraining} onConfigUpdate={handleConfigUpdate} />
              <TrainingControl
                isTraining={isTraining}
                onTrainingStart={handleTrainingStart}
                onTrainingStop={handleTrainingStop}
              />
            </div>

            <div className="column">
              <MetricsDisplay trainingActive={isTraining} />
              <GalleryView refreshTrigger={refreshGallery} />
            </div>
          </div>
        </div>
      </main>

      <footer className="app-footer">
        <p>DCGAN Training Server • Real-time Model Monitoring • GPU Accelerated</p>
        <p>For Mac & Windows • Built with PyTorch & React</p>
      </footer>
    </div>
  );
}

export default App;
