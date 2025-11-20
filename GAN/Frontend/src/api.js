/**
 * API service for communicating with DCGAN backend
 */

import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

export const trainingApi = {
  /**
   * Get server health status
   */
  health: async () => {
    const response = await api.get('/health');
    return response.data;
  },

  /**
   * Get device information
   */
  getDeviceInfo: async () => {
    const response = await api.get('/device-info');
    return response.data;
  },

  /**
   * Generate images for a specific animal type
   */
  generateImages: async ({ animal_type = 'cat', num_images = 16 } = {}) => {
    console.log('Calling generate with:', { animal_type, num_images });
    const response = await api.post('/generate', {
      animal_type,
      num_images: parseInt(num_images, 10),
    });
    console.log('Generate response:', response.data);
    return response.data;
  },

  /**
   * Get available models
   */
  getAvailableModels: async () => {
    const response = await api.get('/available-models');
    return response.data;
  },
};

export default trainingApi;
