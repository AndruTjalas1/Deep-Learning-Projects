import axios from 'axios'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000'

const api = axios.create({
  baseURL: API_URL,
  timeout: 30000,
})

/**
 * Convert blob to base64 string
 * @param {Blob} blob - Image blob
 * @returns {Promise<string>} Base64 encoded string
 */
const blobToBase64 = (blob) => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onloadend = () => resolve(reader.result)
    reader.onerror = reject
    reader.readAsDataURL(blob)
  })
}

/**
 * Recognize a single character from canvas image
 * @param {Blob} imageBlob - Canvas image as blob
 * @returns {Promise} Recognition result with grade and top predictions
 */
export const recognizeCharacter = async (imageBlob) => {
  try {
    const base64Image = await blobToBase64(imageBlob)
    const response = await api.post('/recognize/character', {
      image: base64Image,
    })
    return response.data
  } catch (error) {
    throw error.response?.data || { error: error.message }
  }
}

/**
 * Recognize continuous text (sentence) from canvas image
 * @param {Blob} imageBlob - Canvas image as blob
 * @returns {Promise} Recognition result with per-character details
 */
export const recognizeSentence = async (imageBlob) => {
  try {
    const base64Image = await blobToBase64(imageBlob)
    const response = await api.post('/recognize/sentence', {
      image: base64Image,
    })
    return response.data
  } catch (error) {
    throw error.response?.data || { error: error.message }
  }
}

/**
 * Recognize a single character with specialist model
 * @param {Blob} imageBlob - Canvas image as blob
 * @param {string} characterType - 'digit', 'uppercase', or 'lowercase'
 * @returns {Promise} Recognition result with grade and top predictions
 */
export const recognizeCharacterSpecialist = async (imageBlob, characterType = 'digit') => {
  try {
    const base64Image = await blobToBase64(imageBlob)
    const response = await api.post('/recognize/character/specialist', {
      image: base64Image,
      character_type: characterType
    })
    return response.data
  } catch (error) {
    throw error.response?.data || { error: error.message }
  }
}

/**
 * Recognize handwritten text with auto-routing using character-type classifier
 * @param {Blob} imageBlob - Canvas image as blob
 * @returns {Promise} Recognition result with recognized text and per-character details
 */
export const recognizeText = async (imageBlob) => {
  try {
    const base64Image = await blobToBase64(imageBlob)
    const response = await api.post('/recognize/text', {
      image: base64Image,
    })
    return response.data
  } catch (error) {
    throw error.response?.data || { error: error.message }
  }
}

/**
 * Health check endpoint
 * @returns {Promise} Server status
 */
export const healthCheck = async () => {
  try {
    const response = await api.get('/health')
    return response.data
  } catch (error) {
    throw error.response?.data || { error: error.message }
  }
}

/**
 * Get API information
 * @returns {Promise} API details and endpoints
 */
export const getApiInfo = async () => {
  try {
    const response = await api.get('/api/info')
    return response.data
  } catch (error) {
    throw error.response?.data || { error: error.message }
  }
}

export default api
