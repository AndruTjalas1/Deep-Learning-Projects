import React, { useState } from "react";
import { apiHealth, apiProcess } from "./api";

export default function App() {
  const [text, setText] = useState("");
  const [params, setParams] = useState("{}");
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState(null);

  async function checkHealth() {
    const res = await apiHealth();
    setStatus(res);
  }

  async function run() {
    let parsed = {};
    try {
      parsed = params.trim() ? JSON.parse(params) : {};
    } catch (e) {
      alert("Params must be valid JSON");
      return;
    }
    const res = await apiProcess({ input_text: text, params: parsed });
    setResult(res);
  }

  return (
    <div style={{ maxWidth: 820, margin: "3rem auto", fontFamily: "system-ui, sans-serif" }}>
      <h1>Project UI</h1>

      <section style={{ marginBottom: 24 }}>
        <button onClick={checkHealth}>Check API Health</button>
        {status && (
          <pre style={{ background: "#f5f5f5", padding: 12 }}>
            {JSON.stringify(status, null, 2)}
          </pre>
        )}
      </section>

      <section style={{ marginBottom: 24 }}>
        <label>Input Text</label>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={5}
          style={{ width: "100%" }}
          placeholder="Type input your MethodFiles expect..."
        />

        <label style={{ display: "block", marginTop: 12 }}>
          Params (JSON)
        </label>
        <textarea
          value={params}
          onChange={(e) => setParams(e.target.value)}
          rows={5}
          style={{ width: "100%" }}
          placeholder='{"option": true, "threshold": 0.7}'
        />

        <div style={{ marginTop: 12 }}>
          <button onClick={run}>Run</button>
        </div>
      </section>

      <section>
        <h2>Result</h2>
        <pre style={{ background: "#f5f5f5", padding: 12, minHeight: 120 }}>
          {result ? JSON.stringify(result, null, 2) : "No result yet."}
        </pre>
      </section>
    </div>
  );
}
