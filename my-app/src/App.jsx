// my-app/src/App.jsx
import React from "react";
import Navbar from "./components/Navbar";
import "./index.css";
import "./index.css";

// Read API base from Vite env in prod; localhost in dev as fallback
const API_BASE = (import.meta?.env?.VITE_API_BASE ?? "http://localhost:8000").replace(/\/+$/, "");

// External Streamlit target for Projects 1–3
const STREAMLIT_URL = "https://cst-435-bxasfo3v8izfkfavktqpew.streamlit.app/";

const PROJECTS = [
  { id: 1, title: "Interior Designer", subtitle: "Project 1", slug: "project-1" },
  { id: 2, title: "Sea Animal Identifier", subtitle: "Project 2", slug: "project-2" },
  { id: 3, title: "Review Analysis", subtitle: "Project 3", slug: "project-3" },
  { id: 4, title: "Story Generator", subtitle: "Project 4", slug: "project-4" },
  { id: 5, title: "Cat and Dog Image Generator", subtitle: "Project 5", slug: "project-5" },
  { id: 6, title: "Handwriting Training", subtitle: "Project 6", slug: "project-6" },
];

export default function App() {

  const handleLaunch = (slug, id) => {
    if (id >= 1 && id <= 3) {
      window.location.href = STREAMLIT_URL;
      return;
    }

    if (id === 4) {
      // RNN app
      window.location.href = "/rnn/";
      return;
    }

    if (id === 5) {
      // GAN app
      window.location.href = "/gan/";
      return;
    }

    if (id === 6) {
      // ✅ FIXED: Launch CNN project to /dnp/
      window.location.href = "/dnp/";
      return;
    }

    // fallback
    window.location.href = `/project/${slug}`;
  };

  const handleDetails = (id, title) => {
    if (id >= 1 && id <= 3) {
      alert(`${title}\nStreamlit → Vercel + Railway conversion coming soon.`);
      return;
    }
    if (id === 4) {
      alert(`${title}\nLSTM text-gen trained on large corpora with FastAPI + React.`);
      return;
    }
    if (id === 5) {
      alert(`${title}\nDCGAN for cats & dogs with full-stack deployment.`);
      return;
    }
    if (id === 6) {
      alert(`${title}\nCNN handwriting recognition; segmented letters and confidence scores.`);
      return;
    }

    alert(`${title}\nDetails coming soon.`);
  };

  return (
    <div className="page">
      <Navbar />
      <header className="header">
        <h1 className="title">CST-435</h1>
        <p className="subtitle">Select a base project to get started</p>
      </header>

      <section className="grid">
        {PROJECTS.map(({ id, title, subtitle, slug }) => (
          <article key={id} className="card" tabIndex={0}>
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
        <a className="link" onClick={() => handleLaunch("dnp", 6)} >
          Load /dnp/ Test
        </a>
      </footer>
    </div>
  );
}
