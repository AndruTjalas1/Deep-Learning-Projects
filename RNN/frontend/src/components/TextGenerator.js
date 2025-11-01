// RNN/frontend/src/components/TextGenerator.js
import React, { useState } from "react";
import { rnnApi as api } from "../services/rnnApi";

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
    <div className="card">
      <h2>Text Generator</h2>

      <form onSubmit={handleGenerate} className="form">
        <label>Seed Text:</label>
        <input
          value={seed}
          onChange={(e) => setSeed(e.target.value)}
          placeholder="Enter a starting phrase…"
        />

        <div className="row">
          <div className="col">
            <label>Words to Generate:</label>
            <input
              type="number"
              min={1}
              max={200}
              value={words}
              onChange={(e) => setWords(Math.max(1, Math.min(200, Number(e.target.value))))}
            />
          </div>

          <div className="col">
            <label>Temperature: {Number(temperature).toFixed(2)}</label>
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

        <button type="submit" disabled={loading || !seed.trim()}>
          {loading ? "Generating…" : "✨ Generate Text"}
        </button>
      </form>

      {error && <div className="alert error">⚠️ {error}</div>}

      <textarea
        className="output"
        rows={6}
        readOnly
        value={output}
        placeholder="Your generated text will appear here…"
      />
    </div>
  );
}
