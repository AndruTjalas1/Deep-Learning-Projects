import React, { useRef, useEffect } from 'react'
import './Canvas.css'

/**
 * DrawingCanvas Component
 * 
 * Provides a canvas for users to draw handwritten characters or text.
 * Supports:
 * - Real-time drawing with mouse
 * - Touch input support
 * - Clear and undo functionality
 * - Adjustable brush size and color
 * 
 * @param {Object} props
 * @param {function} props.onDraw - Callback when draw is complete
 * @param {string} props.mode - "character" or "sentence"
 * @param {boolean} props.isLoading - Loading state
 */
const Canvas = ({ onDraw, mode = 'character', isLoading = false }) => {
  const canvasRef = useRef(null)
  const [isDrawing, setIsDrawing] = React.useState(false)
  const [context, setContext] = React.useState(null)
  const [imageHistory, setImageHistory] = React.useState([])
  const [brushSize, setBrushSize] = React.useState(3)

  // Initialize canvas
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    // Set canvas size
    canvas.width = canvas.offsetWidth
    canvas.height = canvas.offsetHeight

    // Get context
    const ctx = canvas.getContext('2d')
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
    ctx.lineWidth = brushSize

    // Set background
    ctx.fillStyle = '#ffffff'
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    // Save initial state
    saveCanvasState()
    setContext(ctx)
  }, [])

  // Update brush size
  useEffect(() => {
    if (context) {
      context.lineWidth = brushSize
    }
  }, [brushSize, context])

  const saveCanvasState = () => {
    const canvas = canvasRef.current
    if (canvas) {
      setImageHistory((prev) => [...prev, canvas.toDataURL()])
    }
  }

  const startDrawing = (e) => {
    if (isLoading) return
    setIsDrawing(true)
    const { offsetX, offsetY } = getCoordinates(e)
    context.beginPath()
    context.moveTo(offsetX, offsetY)
  }

  const draw = (e) => {
    if (!isDrawing || isLoading) return
    const { offsetX, offsetY } = getCoordinates(e)
    context.lineTo(offsetX, offsetY)
    context.stroke()
  }

  const stopDrawing = () => {
    setIsDrawing(false)
    context.closePath()
  }

  const getCoordinates = (e) => {
    const canvas = canvasRef.current
    const rect = canvas.getBoundingClientRect()

    if (e.touches) {
      return {
        offsetX: e.touches[0].clientX - rect.left,
        offsetY: e.touches[0].clientY - rect.top,
      }
    }

    return {
      offsetX: e.clientX - rect.left,
      offsetY: e.clientY - rect.top,
    }
  }

  const clearCanvas = () => {
    const canvas = canvasRef.current
    context.fillStyle = '#ffffff'
    context.fillRect(0, 0, canvas.width, canvas.height)
    setImageHistory([canvasRef.current.toDataURL()])
  }

  const undo = () => {
    if (imageHistory.length > 1) {
      const newHistory = imageHistory.slice(0, -1)
      setImageHistory(newHistory)
      const image = new Image()
      image.src = newHistory[newHistory.length - 1]
      image.onload = () => {
        const canvas = canvasRef.current
        context.clearRect(0, 0, canvas.width, canvas.height)
        context.drawImage(image, 0, 0)
      }
    }
  }

  const submit = async () => {
    const canvas = canvasRef.current
    
    // Find bounding box of drawn content
    const imageData = context.getImageData(0, 0, canvas.width, canvas.height)
    const data = imageData.data
    let minX = canvas.width, minY = canvas.height, maxX = 0, maxY = 0
    let hasPixels = false
    
    // Scan for non-white pixels
    for (let i = 0; i < data.length; i += 4) {
      const r = data[i], g = data[i + 1], b = data[i + 2]
      // Check if pixel is not white (not background)
      if (!(r === 255 && g === 255 && b === 255)) {
        hasPixels = true
        const pixelIndex = i / 4
        const x = pixelIndex % canvas.width
        const y = Math.floor(pixelIndex / canvas.width)
        
        minX = Math.min(minX, x)
        minY = Math.min(minY, y)
        maxX = Math.max(maxX, x)
        maxY = Math.max(maxY, y)
      }
    }
    
    // If no drawn content, alert user
    if (!hasPixels) {
      alert('Please draw something first!')
      return
    }
    
    // Add padding around the content
    const padding = 10
    minX = Math.max(0, minX - padding)
    minY = Math.max(0, minY - padding)
    maxX = Math.min(canvas.width, maxX + padding)
    maxY = Math.min(canvas.height, maxY + padding)
    
    // Create a cropped canvas
    const cropWidth = maxX - minX
    const cropHeight = maxY - minY
    const croppedCanvas = document.createElement('canvas')
    croppedCanvas.width = cropWidth
    croppedCanvas.height = cropHeight
    
    const croppedCtx = croppedCanvas.getContext('2d')
    croppedCtx.drawImage(canvas, minX, minY, cropWidth, cropHeight, 0, 0, cropWidth, cropHeight)
    
    // Convert cropped canvas to blob
    croppedCanvas.toBlob((blob) => {
      onDraw(blob)
    }, 'image/png')
  }

  return (
    <div className="canvas-container">
      <div className="canvas-header">
        <h2>
          {mode === 'character'
            ? '✏️ Draw a Character or Digit'
            : '✏️ Write a Sentence'}
        </h2>
        <p className="info-text">
          {mode === 'character'
            ? 'Draw a single letter (A-Z), number (0-9), or symbol on the canvas below.'
            : 'Write a sentence or word. The system will segment and recognize each character.'}
        </p>
      </div>

      <div className="canvas-toolbar">
        <div className="brush-control">
          <label htmlFor="brush-size">Brush Size:</label>
          <input
            id="brush-size"
            type="range"
            min="1"
            max="15"
            value={brushSize}
            onChange={(e) => setBrushSize(parseInt(e.target.value))}
            disabled={isLoading}
          />
          <span>{brushSize}px</span>
        </div>
        <div className="canvas-actions">
          <button
            className="btn btn-secondary"
            onClick={undo}
            disabled={imageHistory.length <= 1 || isLoading}
            title="Undo last stroke"
          >
            ↶ Undo
          </button>
          <button
            className="btn btn-secondary"
            onClick={clearCanvas}
            disabled={isLoading}
            title="Clear canvas"
          >
            🗑️ Clear
          </button>
          <button
            className="btn btn-primary"
            onClick={submit}
            disabled={isLoading}
            title="Submit for recognition"
          >
            {isLoading ? (
              <>
                <span className="spinner"></span> Processing...
              </>
            ) : (
              <>✓ Submit</>
            )}
          </button>
        </div>
      </div>

      <canvas
        ref={canvasRef}
        className={`drawing-canvas ${isLoading ? 'loading' : ''}`}
        onMouseDown={startDrawing}
        onMouseMove={draw}
        onMouseUp={stopDrawing}
        onMouseLeave={stopDrawing}
        onTouchStart={startDrawing}
        onTouchMove={draw}
        onTouchEnd={stopDrawing}
      />
    </div>
  )
}

export default Canvas
