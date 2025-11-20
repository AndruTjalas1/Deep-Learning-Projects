import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import trainingApi from '../api';
import './MetricsDisplay.css';

export default function MetricsDisplay({ trainingActive }) {
  const [metrics, setMetrics] = useState(null);
  const [chartData, setChartData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchMetrics = async () => {
      if (!trainingActive) return;
      
      try {
        setLoading(true);
        const data = await trainingApi.getTrainingMetrics();
        setMetrics(data);
        
        // Format data for chart
        const formatted = data.g_losses.map((g_loss, idx) => ({
          epoch: idx + 1,
          g_loss: parseFloat(g_loss.toFixed(4)),
          d_loss: parseFloat(data.d_losses[idx].toFixed(4)),
        }));
        setChartData(formatted);
        setError(null);
      } catch (err) {
        console.error('Error fetching metrics:', err);
        // Don't set error if training hasn't started
        if (err.response?.status !== 400) {
          setError('Failed to load metrics');
        }
      } finally {
        setLoading(false);
      }
    };

    const interval = setInterval(fetchMetrics, 3000);
    fetchMetrics();
    
    return () => clearInterval(interval);
  }, [trainingActive]);

  if (!trainingActive || !metrics) {
    return (
      <div className="metrics-display">
        <h2>📊 Training Metrics</h2>
        <p className="placeholder">Metrics will appear here during training</p>
      </div>
    );
  }

  return (
    <div className="metrics-display">
      <h2>📊 Training Metrics</h2>

      {error && <div className="error-message">{error}</div>}

      <div className="metrics-summary">
        <div className="metric-card">
          <span className="metric-label">Current Epoch</span>
          <span className="metric-value">{metrics.current_epoch + 1} / {metrics.total_epochs}</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">G Loss</span>
          <span className="metric-value metric-g">{metrics.latest_g_loss?.toFixed(4)}</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">D Loss</span>
          <span className="metric-value metric-d">{metrics.latest_d_loss?.toFixed(4)}</span>
        </div>
      </div>

      {chartData.length > 0 && (
        <div className="chart-container">
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis
                dataKey="epoch"
                stroke="rgba(255,255,255,0.7)"
              />
              <YAxis
                stroke="rgba(255,255,255,0.7)"
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'rgba(0,0,0,0.8)',
                  border: '1px solid rgba(255,255,255,0.3)',
                  borderRadius: '4px',
                }}
                labelStyle={{ color: 'white' }}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="g_loss"
                stroke="#82ca9d"
                dot={false}
                isAnimationActive={true}
              />
              <Line
                type="monotone"
                dataKey="d_loss"
                stroke="#ffc658"
                dot={false}
                isAnimationActive={true}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
