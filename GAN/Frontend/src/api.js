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
   * Get current configuration
   */
  getConfig: async () => {
    const response = await api.get('/config');
    return response.data;
  },

  /**
   * Update configuration
   */
  updateConfig: async (params) => {
    const response = await api.post('/config/update', null, { params });
    return response.data;
  },

  /**
   * Start training
   */
  startTraining: async (animalTypes = ['cats', 'dogs'], epochs = null) => {
    const response = await api.post('/train/start', null, {
      params: {
        animal_types: animalTypes.join(','),
        ...(epochs && { epochs }),
      },
    });
    return response.data;
  },

  /**
   * Get training status
   */
  getTrainingStatus: async () => {
    const response = await api.get('/train/status');
    return response.data;
  },

  /**
   * Stop training
   */
  stopTraining: async () => {
    const response = await api.post('/train/stop');
    return response.data;
  },

  /**
   * Generate images
   */
  generateImages: async (numImages = 16) => {
    const response = await api.get('/generate', {
      params: { num_images: numImages },
    });
    return response.data;
  },

  /**
   * List all samples
   */
  listSamples: async () => {
    const response = await api.get('/samples');
    return response.data;
  },

  /**
   * Get sample image URL
   */
  getSampleUrl: (sampleName) => {
    return `${API_BASE_URL}/samples/${sampleName}`;
  },

  /**
   * List all models
   */
  listModels: async () => {
    const response = await api.get('/models');
    return response.data;
  },

  /**
   * Get training metrics
   */
  getTrainingMetrics: async () => {
    const response = await api.get('/training-metrics');
    return response.data;
  },
};

export default trainingApi;
