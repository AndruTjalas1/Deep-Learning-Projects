import React, { useState, useEffect } from 'react';
import { FiAlertCircle, FiCheckCircle } from 'react-icons/fi';
import Navbar from './components/Navbar.jsx';
import TextGenerator from './components/TextGenerator.jsx';
import ModelInfo from './components/ModelInfo.jsx';
import { rnnApi as api } from './services/rnnApi';
import './App.css';
import './styles/Navbar.css';

function App() {
  const [isConnected, setIsConnected] = useState(false);
  const [connectionError, setConnectionError] = useState('');

  useEffect(() => {
    checkConnection();
    const interval = setInterval(checkConnection, 30000);
    return () => clearInterval(interval);
  }, []);

  const checkConnection = async () => {
    try {
      const h = await api.health();
      setIsConnected(h?.status === 'healthy' || h?.model_loaded === true);
      setConnectionError('');
    } catch (error) {
      setIsConnected(false);
      setConnectionError('Cannot connect to backend API. Make sure the server is reachable via /api');
    }
  };

  return (
    <div className="app">
      <Navbar />
      {/* <header className="app-header">
        <div className="header-content">
          <h1>RNN Text Generator</h1>
          <p>Generate creative text using LSTM Neural Networks</p>
        </div>
        <div className={`status-indicator ${isConnected ? 'connected' : 'disconnected'}`}>
          {isConnected ? (
            <><FiCheckCircle className="status-dot" /><span className="status-text">Connected</span></>
          ) : (
            <><FiAlertCircle className="status-dot" /><span className="status-text">Disconnected</span></>
          )}
        </div>
      </header> */}

      <main className="app-main">
        {connectionError && (
          <div className="connection-alert">
            <FiAlertCircle className="alert-icon" />
            <div className="alert-content">
              <strong>Connection Issue</strong>
              <p>{connectionError}</p>
              <button onClick={checkConnection}>Retry Connection</button>
            </div>
          </div>
        )}

        <div className="content-grid">
          <section className="main-section">
            <TextGenerator />
          </section>

          <aside className="info-section">
            <ModelInfo />
          </aside>
        </div>
      </main>

      <footer className="app-footer">
        <p>RNN Text Generator • CST-435</p>
      </footer>
    </div>
  );
}

export default App;
