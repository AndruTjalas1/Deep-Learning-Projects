// my-app/src/App.jsx
import React from "react";
import "./index.css";

// Read API base from Vite env in prod; localhost in dev as fallback
const API_BASE = (import.meta?.env?.VITE_API_BASE ?? "http://localhost:8000").replace(/\/+$/, "");

// External Streamlit target for Projects 1–4
const STREAMLIT_URL = "https://cst-435-bxasfo3v8izfkfavktqpew.streamlit.app/";

const PROJECTS = [
  { id: 1, title: "Perceptron", subtitle: "Project 1", emoji: "🧠", slug: "project-1" },
  { id: 2, title: "Artificial Neural Network (ANN)", subtitle: "Project 2", emoji: "🧩", slug: "project-2" },
  { id: 3, title: "Neural Network", subtitle: "Project 3", emoji: "🕸️", slug: "project-3" },
  { id: 4, title: "NLP Application", subtitle: "Project 4", emoji: "🗣️", slug: "project-4" },
  { id: 5, title: "Recurrent Neural Network", subtitle: "Project 5", emoji: "🎨", slug: "project-5" }, // RNN stays as-is
  { id: 6, title: "Deep Neural Network Performance", subtitle: "Project 6", emoji: "⚙️", slug: "project-6" },
  { id: 7, title: "GAN-Based Application", subtitle: "Project 7", emoji: "🧪", slug: "project-7" },
  { id: 8, title: "Deep Neural Network Project", subtitle: "Project 8", emoji: "☁️", slug: "project-8" },
];

export default function App() {
  const handleLaunch = (slug, id) => {
    if (id >= 1 && id <= 4) {
      // Projects 1–4 go to Streamlit app
      window.location.href = STREAMLIT_URL;
      return;
    }
    if (id === 5) {
      // Project 5 (RNN) goes to the bundled page
      window.location.href = "/rnn/";
      return;
    }
    // Projects 6–8 keep internal route (placeholder)
    window.location.href = `/project/${slug}`;
  };

  const handleDetails = (id, title) => {
    if (id >= 1 && id <= 4) {
      alert(`${title}\nStreamlit → Vercel + Railway conversion coming soon.`);
      return;
    }
    if (id === 5) {
      alert(`${title}\nThis RNN app is live. Explore it on the /rnn/ page.`);
      return;
    }
    // 6–8
    alert(`${title}\nDetails coming soon.`);
  };

  return (
    <div className="page">
      <header className="header">
        <h1 className="title">CST-435</h1>
        <p className="subtitle">Select a base project to get started</p>
      </header>

      <section className="grid">
        {PROJECTS.map(({ id, title, subtitle, emoji, slug }) => (
          <article key={id} className="card" tabIndex={0}>
            <div className="badge">{emoji}</div>
            <h2 className="cardTitle">{title}</h2>
            <p className="cardText">{subtitle}</p>
            <div className="actions">
              <button className="btn primary" onClick={() => handleLaunch(slug, id)}>
                Launch
              </button>
              <button className="btn ghost" onClick={() => handleDetails(id, title)}>
                Details
              </button>
            </div>
          </article>
        ))}
      </section>

      <footer className="footer">
        {/* Opens your Railway health endpoint explicitly */}
        <a
          className="link"
          href={`${API_BASE}/api/health`}
          target="_blank"
          rel="noreferrer"
        >
          API Health
        </a>
        {/* (Optional) show which API base is active */}
        {/* <span className="muted" style={{ marginLeft: 12 }}>API: {API_BASE}</span> */}
      </footer>
    </div>
  );
}
