import React from 'react'
import './ResultDisplay.css'

/**
 * Character Recognition Result Component
 * Displays the recognized character with grade and confidence
 */
export const CharacterResult = ({ result, loading }) => {
  if (loading) {
    return (
      <div className="result-container loading">
        <span className="spinner"></span>
        <p>Analyzing your handwriting...</p>
      </div>
    )
  }

  if (!result) return null

  if (result.error) {
    return <div className="error-message">{result.error}</div>
  }

  const gradeInfo = result.grade_info
  const gradeColor = {
    A: '#16a34a',
    B: '#2563eb',
    C: '#ea580c',
    D: '#f59e0b',
    F: '#dc2626',
  }[gradeInfo.grade]

  return (
    <div className="result-container character-result">
      <div className="success-message">✓ Recognition Complete</div>

      <div className="character-display">
        <div
          className="predicted-character"
          style={{ borderColor: gradeColor }}
        >
          {gradeInfo.predicted_character}
        </div>

        <div className="grade-display" style={{ color: gradeColor }}>
          <div className="grade-letter">{gradeInfo.grade}</div>
          <div className="grade-label">Grade</div>
        </div>
      </div>

      <div className="grade-details">
        <div className="detail-row">
          <span className="label">Character:</span>
          <span className="value">{gradeInfo.predicted_character}</span>
        </div>
        <div className="detail-row">
          <span className="label">Confidence:</span>
          <span className="value">{(gradeInfo.confidence * 100).toFixed(1)}%</span>
        </div>
        <div className="detail-row">
          <span className="label">Feedback:</span>
          <span className="value">{gradeInfo.feedback}</span>
        </div>
      </div>

      <div className="top-predictions">
        <h4>Top Predictions:</h4>
        <div className="predictions-list">
          {result.top_predictions.map((pred, idx) => (
            <div key={idx} className="prediction-item">
              <span className="char">{pred.character}</span>
              <div className="bar">
                <div
                  className="bar-fill"
                  style={{ width: `${pred.confidence * 100}%` }}
                ></div>
              </div>
              <span className="conf">{(pred.confidence * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

/**
 * Sentence Recognition Result Component
 * Displays per-character recognition results
 */
export const SentenceResult = ({ result, loading }) => {
  if (loading) {
    return (
      <div className="result-container loading">
        <span className="spinner"></span>
        <p>Recognizing your handwriting...</p>
      </div>
    )
  }

  if (!result) return null

  if (result.error) {
    return <div className="error-message">{result.error}</div>
  }

  if (!result.text) {
    return (
      <div className="info-box">
        <p>No characters detected. Please write on the canvas and try again.</p>
      </div>
    )
  }

  const gradeColor = (grade) => ({
    A: '#16a34a',
    B: '#2563eb',
    C: '#ea580c',
    D: '#f59e0b',
    F: '#dc2626',
  })[grade]

  return (
    <div className="result-container sentence-result">
      <div className="success-message">✓ Text Recognition Complete</div>

      <div className="recognized-text">
        <h3>Recognized Text:</h3>
        <div className="text-output">
          {result.text.split('').map((char, idx) => (
            <span key={idx} className="text-character">
              {char}
            </span>
          ))}
        </div>
      </div>

      <div className="metrics-grid">
        <div className="metric-card">
          <div className="metric-value">{result.num_characters}</div>
          <div className="metric-label">Characters</div>
        </div>
        <div className="metric-card">
          <div className="metric-value">
            {(result.average_confidence * 100).toFixed(1)}%
          </div>
          <div className="metric-label">Avg Confidence</div>
        </div>
        <div className="metric-card">
          <div className="metric-value">
            {(result.success_rate * 100).toFixed(1)}%
          </div>
          <div className="metric-label">Success Rate</div>
        </div>
      </div>

      <div className="character-breakdown">
        <h4>Character-by-Character Analysis:</h4>
        <div className="characters-grid">
          {result.characters.map((charInfo, idx) => (
            <div
              key={idx}
              className="char-card"
              style={{ borderTopColor: gradeColor(charInfo.grade) }}
            >
              <div className="char-main">
                <span className="char-display">{charInfo.character}</span>
                <span
                  className="char-grade"
                  style={{ backgroundColor: gradeColor(charInfo.grade) }}
                >
                  {charInfo.grade}
                </span>
              </div>
              <div className="char-info">
                <span className="confidence">
                  {(charInfo.confidence * 100).toFixed(0)}%
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="info-box">
        <p>
          <strong>Pro Tip:</strong> Characters with A or B grades are recognized
          with high confidence. Try to write more clearly for D and F grades.
        </p>
      </div>
    </div>
  )
}

export const ResultDisplay = ({ result, loading, mode }) => {
  if (mode === 'character') {
    return <CharacterResult result={result} loading={loading} />
  } else {
    return <SentenceResult result={result} loading={loading} />
  }
}

export default ResultDisplay
