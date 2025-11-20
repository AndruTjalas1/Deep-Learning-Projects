import React, { useState, useEffect } from 'react';
import trainingApi from '../api';
import './GalleryView.css';

export default function GalleryView({ refreshTrigger }) {
  const [samples, setSamples] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedImage, setSelectedImage] = useState(null);

  useEffect(() => {
    fetchSamples();
  }, [refreshTrigger]);

  const fetchSamples = async () => {
    try {
      setLoading(true);
      const data = await trainingApi.listSamples();
      setSamples(data.samples || []);
      setError(null);
    } catch (err) {
      console.error('Error fetching samples:', err);
      setError('Failed to load samples');
    } finally {
      setLoading(false);
    }
  };

  const handleRefresh = async () => {
    await fetchSamples();
  };

  // Get most recent samples (last 12)
  const recentSamples = samples.slice(-12).reverse();

  return (
    <div className="gallery-view">
      <div className="gallery-header">
        <h2>🎨 Generated Samples</h2>
        <button onClick={handleRefresh} disabled={loading} className="refresh-btn">
          {loading ? 'Loading...' : '🔄 Refresh'}
        </button>
      </div>

      {error && <div className="error-message">{error}</div>}

      <div className="gallery-info">
        <p>Total samples: {samples.length}</p>
      </div>

      {recentSamples.length === 0 ? (
        <div className="gallery-empty">
          <p>No samples generated yet. Start training to see generated images!</p>
        </div>
      ) : (
        <div className="gallery-grid">
          {recentSamples.map((sample, idx) => (
            <div
              key={idx}
              className="gallery-item"
              onClick={() => setSelectedImage(sample)}
            >
              <img
                src={trainingApi.getSampleUrl(sample)}
                alt={`Sample ${sample}`}
                onError={(e) => {
                  e.target.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="200" height="200"%3E%3Crect fill="%23333" width="200" height="200"/%3E%3C/svg%3E';
                }}
              />
              <div className="gallery-item-info">
                <p>{sample.slice(-20)}</p>
              </div>
            </div>
          ))}
        </div>
      )}

      {selectedImage && (
        <div className="modal" onClick={() => setSelectedImage(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close" onClick={() => setSelectedImage(null)}>×</button>
            <img src={trainingApi.getSampleUrl(selectedImage)} alt={selectedImage} />
            <p>{selectedImage}</p>
          </div>
        </div>
      )}
    </div>
  );
}
