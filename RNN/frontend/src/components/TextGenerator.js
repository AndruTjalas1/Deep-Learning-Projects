// RNN/frontend/src/components/TextGenerator.js
import React, { useState } from "react";
import { rnnApi as api } from "../services/rnnApi";
import "./TextGenerator.css";

export default function TextGenerator() {
  const [seed, setSeed] = useState("the");
  const [words, setWords] = useState(20);
  const [temperature, setTemperature] = useState(1.0);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [output, setOutput] = useState("");

  const handleGenerate = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    setOutput("");

    try {
      const res = await api.generate({
        seed,
        words: Number(words),
        temperature: Number(temperature),
      });
      setOutput(res.generated_text || "");
    } catch (err) {
      // Try to show server-provided error detail if present
      const msg =
        err?.message ||
        (typeof err === "string" ? err : "") ||
        "Generation failed";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="text-generator">
      <h2>💬 Text Generator</h2>

      <form onSubmit={handleGenerate} className="generator-form">
        <div className="form-group">
          <label>Seed Text</label>
          <input
            type="text"
            value={seed}
            onChange={(e) => setSeed(e.target.value)}
            placeholder="Enter a starting phrase…"
          />
        </div>

        <div className="form-row">
          <div className="form-group">
            <label>Words to Generate</label>
            <input
              type="number"
              min={1}
              max={200}
              value={words}
              onChange={(e) => setWords(Math.max(1, Math.min(200, Number(e.target.value))))}
            />
          </div>

          <div className="form-group">
            <label>Temperature: <span className="temp-value">{Number(temperature).toFixed(2)}</span></label>
            <input
              type="range"
              min={0.2}
              max={1.6}
              step={0.05}
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
            />
          </div>
        </div>

        <button type="submit" disabled={loading || !seed.trim()} className="generate-btn">
          {loading ? "⏳ Generating…" : "✨ Generate Text"}
        </button>
      </form>

      {error && <div className="error-message">⚠️ {error}</div>}

      {output && (
        <div className="output-container">
          <h3>Generated Text</h3>
          <div className="generated-text">
            <p>{output}</p>
          </div>
        </div>
      )}
      {!output && !error && (
        <div className="output-container placeholder">
          <h3>Generated Text</h3>
          <textarea
            className="output-textarea"
            rows={6}
            readOnly
            value={output}
            placeholder="Your generated text will appear here…"
          />
        </div>
      )}
    </div>
  );
}
