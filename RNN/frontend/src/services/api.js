// API service for communicating with the backend
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const api = {
  // Health check
  healthCheck: async () => {
    try {
      const response = await apiClient.get('/');
      return response.data;
    } catch (error) {
      console.error('Health check failed:', error);
      throw error;
    }
  },

  // Get model info
  getModelInfo: async () => {
    try {
      const response = await apiClient.get('/model/info');
      return response.data;
    } catch (error) {
      console.error('Failed to get model info:', error);
      throw error;
    }
  },

  // Generate text
  generateText: async (seedText, numWords = 50, temperature = 1.0) => {
    try {
      const response = await apiClient.post('/generate', {
        seed_text: seedText,
        num_words: numWords,
        temperature: temperature,
      });
      return response.data;
    } catch (error) {
      console.error('Failed to generate text:', error);
      throw error;
    }
  },

  // Get stats
  getStats: async () => {
    try {
      const response = await apiClient.get('/stats');
      return response.data;
    } catch (error) {
      console.error('Failed to get stats:', error);
      throw error;
    }
  },

  // Get training history visualization
  getTrainingPlot: async () => {
    try {
      const response = await apiClient.get('/visualizations/training', {
        responseType: 'blob',
      });
      return URL.createObjectURL(response.data);
    } catch (error) {
      console.error('Failed to get training plot:', error);
      throw error;
    }
  },
};

export default api;
