import React, { useState, useRef, useEffect } from 'react';
import './CharacterGradingSpecialist.css';
import { recognizeCharacterSpecialist } from '../api';

export default function CharacterGradingSpecialist() {
    const canvasRef = useRef(null);
    const [characterType, setCharacterType] = useState('digit');
    const [prediction, setPrediction] = useState(null);
    const [loading, setLoading] = useState(false);
    const [hasDrawn, setHasDrawn] = useState(false);
    const [ctx, setCtx] = useState(null);

    // Initialize canvas context on mount
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const context = canvas.getContext('2d');
        
        // Set canvas size
        canvas.width = 600;
        canvas.height = 600;
        
        // Configure drawing properties
        context.strokeStyle = '#000';
        context.lineWidth = 8;
        context.lineCap = 'round';
        context.lineJoin = 'round';
        context.fillStyle = '#fff';
        context.fillRect(0, 0, canvas.width, canvas.height);
        
        setCtx(context);
    }, []);

    // Drawing functions
    const startDrawing = (e) => {
        if (!ctx) return;
        const canvas = canvasRef.current;
        const rect = canvas.getBoundingClientRect();
        
        // Get accurate position accounting for DPI scaling
        const x = (e.clientX - rect.left) * (canvas.width / rect.width);
        const y = (e.clientY - rect.top) * (canvas.height / rect.height);
        
        ctx.beginPath();
        ctx.moveTo(x, y);
    };

    const draw = (e) => {
        if (e.buttons !== 1 || !ctx) return;
        
        const canvas = canvasRef.current;
        const rect = canvas.getBoundingClientRect();
        
        // Get accurate position accounting for DPI scaling
        const x = (e.clientX - rect.left) * (canvas.width / rect.width);
        const y = (e.clientY - rect.top) * (canvas.height / rect.height);
        
        ctx.lineTo(x, y);
        ctx.stroke();
        setHasDrawn(true);
    };

    const clearCanvas = () => {
        if (!ctx) return;
        const canvas = canvasRef.current;
        ctx.fillStyle = '#fff';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        setPrediction(null);
        setHasDrawn(false);
    };

    const submitDrawing = async () => {
        const canvas = canvasRef.current;
        const imageBlob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));

        setLoading(true);
        try {
            const result = await recognizeCharacterSpecialist(imageBlob, characterType);
            setPrediction(result);
        } catch (error) {
            console.error('Error:', error);
            setPrediction({ error: error.error || 'Failed to process image' });
        } finally {
            setLoading(false);
        }
    };

    const getGradeColor = (grade) => {
        const colors = {
            'A': '#27ae60',
            'B': '#2ecc71',
            'C': '#f39c12',
            'D': '#e74c3c',
            'F': '#c0392b'
        };
        return colors[grade] || '#333';
    };



    return (
        <div className="character-grading-specialist">
            {/* Header */}
            <div className="header">
                <h1>Handwriting Character Grading</h1>
                <p>Draw a character and get instant feedback!</p>
            </div>

            {/* Character Type Selection - Button Style */}
            <div className="character-type-selector">
                <label>Select character type:</label>
                <div className="button-group">
                    <button
                        className={`type-button ${characterType === 'digit' ? 'active' : ''}`}
                        onClick={() => setCharacterType('digit')}
                        disabled={loading}
                    >
                        <span className="icon"></span>
                        <span className="label">Digit</span>
                        <span className="range">(0-9)</span>
                    </button>
                    <button
                        className={`type-button ${characterType === 'uppercase' ? 'active' : ''}`}
                        onClick={() => setCharacterType('uppercase')}
                        disabled={loading}
                    >
                        <span className="icon"></span>
                        <span className="label">Uppercase</span>
                        <span className="range">(A-Z)</span>
                    </button>
                    <button
                        className={`type-button ${characterType === 'lowercase' ? 'active' : ''}`}
                        onClick={() => setCharacterType('lowercase')}
                        disabled={loading}
                    >
                        <span className="icon"></span>
                        <span className="label">Lowercase</span>
                        <span className="range">(a-z)</span>
                    </button>
                </div>
            </div>

            {/* Canvas and Controls */}
            <div className="canvas-section">
                <div className="canvas-wrapper">
                    <canvas
                        ref={canvasRef}
                        onMouseDown={startDrawing}
                        onMouseMove={draw}
                        className="drawing-canvas"
                        style={{
                            border: '3px solid #333',
                            cursor: 'crosshair',
                            backgroundColor: '#fff',
                            width: '600px',
                            height: '600px',
                            display: 'block'
                        }}
                    />
                    <p className="canvas-label">Draw your character here</p>
                </div>

                {/* Action Buttons */}
                <div className="button-controls">
                    <button
                        className="btn btn-clear"
                        onClick={clearCanvas}
                        disabled={loading || !hasDrawn}
                    >
                        🗑️ Clear
                    </button>
                    <button
                        className="btn btn-submit"
                        onClick={submitDrawing}
                        disabled={loading || !hasDrawn}
                    >
                        {loading ? '⏳ Analyzing...' : '✓ Submit'}
                    </button>
                </div>
            </div>

            {/* Results Display */}
            {prediction && !prediction.error && (
                <div className="results-section">
                    {/* Main Grade Display */}
                    <div className="grade-display">
                        <div className="grade-badge" style={{ color: getGradeColor(prediction.grade_info.grade) }}>
                            {prediction.grade_info.grade}
                        </div>
                        <div className="prediction-info">
                            <h2>Predicted: <span className="predicted-char">{prediction.predicted_character}</span></h2>
                            <p className="confidence">
                                Confidence: <strong>{(prediction.confidence * 100).toFixed(1)}%</strong>
                            </p>
                        </div>
                    </div>

                    {/* Feedback Message */}
                    <div
                        className="feedback-message"
                        style={{ borderLeftColor: getGradeColor(prediction.grade_info.grade) }}
                    >
                        <p>{prediction.grade_info.feedback}</p>
                    </div>

                    {/* Top Predictions */}
                    {prediction.top_predictions.length > 1 && (
                        <div className="alternatives-section">
                            <h3>Other Possibilities:</h3>
                            <div className="alternatives-grid">
                                {prediction.top_predictions.slice(1).map((pred, idx) => (
                                    <div key={idx} className="alternative-item">
                                        <div className="alt-char">{pred.character}</div>
                                        <div className="alt-confidence">
                                            {(pred.confidence * 100).toFixed(1)}%
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Debug Info */}
                    <div className="debug-info">
                        <small>Model: {prediction.character_type} • Device: {prediction.inference_device}</small>
                    </div>
                </div>
            )}

            {/* Error Display */}
            {prediction?.error && (
                <div className="error-message">
                    <p>❌ {prediction.error}</p>
                </div>
            )}
        </div>
    );
}
