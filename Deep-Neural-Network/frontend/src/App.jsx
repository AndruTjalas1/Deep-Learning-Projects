import React, { useState, useEffect } from 'react'
import { FiActivity, FiAlertCircle } from 'react-icons/fi'
import Navbar from './components/Navbar'
import Canvas from './components/Canvas'
import CharacterGradingSpecialist from './components/CharacterGradingSpecialist'
import { ResultDisplay } from './components/ResultDisplay'
import { recognizeCharacter, recognizeSentence, healthCheck } from './api'
import './App.css'
import './styles/Navbar.css'

function App() {
  const [activeTab, setActiveTab] = useState('character')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [serverStatus, setServerStatus] = useState(null)

  // Check server health on mount
  useEffect(() => {
    const checkServer = async () => {
      try {
        const status = await healthCheck()
        setServerStatus(status)
      } catch (err) {
        setServerStatus({ error: 'Server connection failed' })
      }
    }
    checkServer()
  }, [])

  const handleDraw = async (imageBlob) => {
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      let apiResult

      if (activeTab === 'character') {
        apiResult = await recognizeCharacter(imageBlob)
      } else {
        apiResult = await recognizeSentence(imageBlob)
      }

      setResult(apiResult)
    } catch (err) {
      setError(
        err.error ||
          err.message ||
          'An error occurred during recognition'
      )
      console.error('Recognition error:', err)
    } finally {
      setLoading(false)
    }
  }

  const switchTab = (tab) => {
    setActiveTab(tab)
    setResult(null)
    setError(null)
  }

  return (
    <div className="app">
      <Navbar />
      {/* <header className="app-header">
        <div className="container">
          <div className="header-content">
            <h1>Handwriting Recognition System</h1>
            <p className="tagline">
              Draw letters, numbers, or sentences to see AI recognition in action
            </p>
            <div className="server-status">
              {serverStatus?.error ? (
                <span className="status error">
                  <FiAlertCircle /> Server Offline
                </span>
              ) : serverStatus ? (
                <span className="status online">
                  <FiActivity /> Server Online
                </span>
              ) : (
                <span className="status loading">
                  Checking...
                </span>
              )}
            </div>
          </div>
        </div>
      </header> */}

      <main className="app-main">
        <div className="container">
          <div className="tabs">
            <button
              className={`tab-btn ${activeTab === 'character' ? 'active' : ''}`}
              onClick={() => switchTab('character')}
            >
              <span className="tab-icon"></span>
              Character Recognition
            </button>
            <button
              className={`tab-btn ${activeTab === 'sentence' ? 'active' : ''}`}
              onClick={() => switchTab('sentence')}
            >
              <span className="tab-icon"></span>
              Text to Speech
            </button>
          </div>

          {error && (
            <div className="error-message">
              <strong>Error:</strong> {error}
            </div>
          )}

          {activeTab === 'character' ? (
            <CharacterGradingSpecialist />
          ) : (
            <>
              <Canvas
                onDraw={handleDraw}
                mode={activeTab}
                isLoading={loading}
              />
              <ResultDisplay
                result={result}
                loading={loading}
                mode={activeTab}
              />
            </>
          )}
        </div>
      </main>

      <footer className="app-footer">
        <p>Deep Neural Network • CST-435</p>
      </footer>
    </div>
  )
}

export default App
