import { useEffect, useState, useRef } from "react";
import "./App.css";

const WS_URL = "ws://localhost:8000/ws";
const API_URL = "http://localhost:8000";

export default function App() {
  const [dataset, setDataset] = useState("mnist");
  const [classFilter, setClassFilter] = useState("");
  const [epochs, setEpochs] = useState(10);
  const [batchSize, setBatchSize] = useState(64);
  const [learningRate, setLearningRate] = useState(0.0002);
  const [device, setDevice] = useState("mps");

  const [isTraining, setIsTraining] = useState(false);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [logMessages, setLogMessages] = useState([]);
  const [sampleImage, setSampleImage] = useState(null);

  const [activeTab, setActiveTab] = useState("logs");

  const wsRef = useRef(null);

  useEffect(() => {
    connectWS();
  }, []);

  const connectWS = () => {
    wsRef.current = new WebSocket(WS_URL);

    wsRef.current.onmessage = (event) => {
      const msg = JSON.parse(event.data);

      if (msg.type === "batch_update") {
        addLog(`Epoch ${msg.epoch} | Batch ${msg.batch} — G:${msg.metrics.loss_g.toFixed(3)} D:${msg.metrics.loss_d.toFixed(3)}`);
      }

      if (msg.type === "epoch_complete") {
        setCurrentEpoch(msg.epoch);
        addLog(`Epoch ${msg.epoch} complete`);
        if (msg.sample_image) setSampleImage(msg.sample_image);
      }

      if (msg.type === "training_complete") {
        addLog("Training finished.");
        setIsTraining(false);
      }
    };

    wsRef.current.onclose = () => {
      setTimeout(connectWS, 1000);
    };
  };

  const addLog = (text) => {
    setLogMessages((prev) => [...prev, text]);
  };

  const startTraining = async () => {
    setIsTraining(true);
    setLogMessages([]);

    const res = await fetch(`${API_URL}/start_training`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset,
        epochs,
        batch_size: batchSize,
        learning_rate: learningRate,
        device,
        class_filter: classFilter || null,
      }),
    });

    const data = await res.json();
    addLog("Training started...");
  };

  const stopTraining = async () => {
    await fetch(`${API_URL}/stop_training`, { method: "POST" });
    addLog("Training stopped by user.");
  };

  const saveModel = async () => {
    const res = await fetch(`${API_URL}/save_model`, { method: "POST" });
    const data = await res.json();
    addLog("Model saved.");
  };

  const loadModel = async () => {
    const res = await fetch(`${API_URL}/load_model`, { method: "POST" });
    const data = await res.json();
    addLog("Model loaded.");
  };

  return (
    <div className="app-container">
      <h1 className="title">DCGAN Training Dashboard</h1>

      {/* Controls Panel */}
      <div className="panel">
        <div className="panel-section">
          <label>Dataset</label>
          <select value={dataset} onChange={(e) => setDataset(e.target.value)}>
            <option value="mnist">MNIST</option>
            <option value="fashion_mnist">Fashion MNIST</option>
            <option value="cifar10">CIFAR-10</option>
          </select>
        </div>

        {dataset === "cifar10" && (
          <div className="panel-section">
            <label>CIFAR-10 Class (optional)</label>
            <input
              placeholder="cat, dog, car, truck..."
              value={classFilter}
              onChange={(e) => setClassFilter(e.target.value)}
            />
          </div>
        )}

        <div className="panel-section">
          <label>Epochs</label>
          <input
            type="number"
            value={epochs}
            onChange={(e) => setEpochs(Number(e.target.value))}
          />
        </div>

        <div className="panel-section">
          <label>Batch Size</label>
          <input
            type="number"
            value={batchSize}
            onChange={(e) => setBatchSize(Number(e.target.value))}
          />
        </div>

        <div className="panel-section">
          <label>Device</label>
          <select value={device} onChange={(e) => setDevice(e.target.value)}>
            <option value="mps">MPS (Mac GPU)</option>
            <option value="cuda">CUDA</option>
            <option value="cpu">CPU</option>
          </select>
        </div>

        <div className="button-group">
          {!isTraining ? (
            <button className="btn start" onClick={startTraining}>Start</button>
          ) : (
            <button className="btn stop" onClick={stopTraining}>Stop</button>
          )}
          <button className="btn save" onClick={saveModel}>Save Model</button>
          <button className="btn load" onClick={loadModel}>Load Model</button>
        </div>
      </div>

      {/* Tabs */}
      <div className="tabs">
        <button className={activeTab === "logs" ? "active" : ""} onClick={() => setActiveTab("logs")}>
          Logs
        </button>
        <button className={activeTab === "samples" ? "active" : ""} onClick={() => setActiveTab("samples")}>
          Samples
        </button>
      </div>

      {/* Content Area */}
      <div className="content-area">
        {activeTab === "logs" && (
          <div className="logs">
            {logMessages.map((msg, i) => (
              <div key={i} className="log-entry">
                {msg}
              </div>
            ))}
          </div>
        )}

        {activeTab === "samples" && (
          <div className="samples">
            {sampleImage ? (
              <img
                className="sample-img"
                src={`data:image/png;base64,${sampleImage}`}
                alt="sample"
              />
            ) : (
              <div className="no-sample">No image yet</div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
