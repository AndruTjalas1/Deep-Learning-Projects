import React, { useState, useEffect } from 'react';
import { FiCpu, FiGitBranch, FiChevronRight, FiAlertCircle } from 'react-icons/fi';
import Navbar from './components/Navbar';
import trainingApi from './api';
import './App.css';
import './styles/Navbar.css';

function App() {
  const [selectedAnimal, setSelectedAnimal] = useState('cat');
  const [numImages, setNumImages] = useState(16);
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedImages, setGeneratedImages] = useState([]);
  const [deviceInfo, setDeviceInfo] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchDeviceInfo = async () => {
      try {
        const info = await trainingApi.getDeviceInfo();
        setDeviceInfo(info);
      } catch (err) {
        console.error('Error fetching device info:', err);
      }
    };

    fetchDeviceInfo();
  }, []);

  const handleGenerate = async () => {
    setIsGenerating(true);
    setError(null);
    console.log('Generating with numImages:', numImages, 'type:', typeof numImages);
    try {
      const data = await trainingApi.generateImages({
        animal_type: selectedAnimal,
        num_images: numImages,
      });
      
      if (data.images) {
        setGeneratedImages(data.images);
      }
    } catch (err) {
      setError(`Error generating images: ${err.message}`);
      console.error('Generation error:', err);
    } finally {
      setIsGenerating(false);
    }
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
      <Navbar />
      {/* <header className="app-header">
        <h1>DCGAN Image Generator</h1>
        <p>Generate AI-created cat and dog images</p>
        <div className="device-status">
          <FiCpu /> Device: {getDeviceString()}
        </div>
      </header> */}

      <main className="app-main">
        <div className="container">
          <div className="control-panel">
            <h2>Generate Images</h2>
            
            <div className="control-group">
              <label>Select Animal Type:</label>
              <div className="button-group">
                <button
                  className={`animal-button ${selectedAnimal === 'cat' ? 'active' : ''}`}
                  onClick={() => setSelectedAnimal('cat')}
                  disabled={isGenerating}
                >
                  Cat Images
                </button>
                <button
                  className={`animal-button ${selectedAnimal === 'dog' ? 'active' : ''}`}
                  onClick={() => setSelectedAnimal('dog')}
                  disabled={isGenerating}
                >
                  Dog Images
                </button>
              </div>
            </div>

            <div className="control-group">
              <label htmlFor="numImages">Number of Images:</label>
              <div className="input-group">
                <input
                  id="numImages"
                  type="range"
                  min="1"
                  max="64"
                  value={numImages}
                  onChange={(e) => setNumImages(parseInt(e.target.value))}
                  disabled={isGenerating}
                />
                <span className="number-display">{numImages}</span>
              </div>
            </div>

            <button
              className="generate-button"
              onClick={handleGenerate}
              disabled={isGenerating}
            >
              <FiChevronRight />
              {isGenerating ? 'Generating...' : 'Generate Images'}
            </button>

            {error && <div className="error-message"><FiAlertCircle /> {error}</div>}
          </div>

          {generatedImages.length > 0 && (
            <div className="gallery">
              <h2>Generated Images ({generatedImages.length})</h2>
              <div className="image-grid">
                {generatedImages.map((imgData, idx) => (
                  <div key={idx} className="image-item">
                    <img
                      src={`data:image/png;base64,${imgData}`}
                      alt={`Generated ${selectedAnimal} ${idx + 1}`}
                    />
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </main>

      <footer className="app-footer">
        <p>DCGAN Image Generator • CST-435</p>
      </footer>
    </div>
  );
}

export default App;
