// my-app/src/App.jsx
import React from "react";
import Navbar from "./components/Navbar";
import "./index.css";

// Read API base from Vite env in prod; localhost in dev as fallback
const API_BASE = (import.meta?.env?.VITE_API_BASE ?? "http://localhost:8000").replace(/\/+$/, "");

// External Streamlit target for Projects 1–4
const STREAMLIT_URL = "https://cst-435-bxasfo3v8izfkfavktqpew.streamlit.app/";

const PROJECTS = [
  { id: 1, title: "Interior Designer", subtitle: "Project 1", emoji: "", slug: "project-1" },
  { id: 2, title: "Sea Animal Identifier", subtitle: "Project 2", emoji: "", slug: "project-3" },
  { id: 3, title: "Review Analysis", subtitle: "Project 3", emoji: "", slug: "project-4" },
  { id: 4, title: "Story Generator", subtitle: "Project 4", emoji: "", slug: "project-5" },
  { id: 5, title: "Cat and Dog Image Generator", subtitle: "Project 5", emoji: "", slug: "project-7" },
  { id: 6, title: "Handwriting Training", subtitle: "Project 6", emoji: "", slug: "project-8" },
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
      // ✅ GAN app
      window.location.href = "/gan/";
      return;
    }

    if (id === 6) {
      // Placeholder for CNN app
      window.location.href = "/dnp/";
      return;
    }

    // Projects 6 & 8 remain placeholders
    window.location.href = `/project/${slug}`;
  };

  const handleDetails = (id, title) => {
    if (id >= 1 && id <= 3) {
      alert(`${title}\nStreamlit → Vercel + Railway conversion coming soon.`);
      return;
    }
    if (id === 4) {
      alert(`${title}\nLSTM-based text generation system using deep learning. Trained on text data and deployed with FastAPI backend and React frontend.`);
      return;
    }
    if (id === 6) {
      alert(`${title}\nCNN-based handwriting recognition with character segmentation. Recognizes individual characters and continuous text with confidence scoring.`);
      return;
    }
    if (id === 5) {
      alert(`${title}\nDCGAN system for generating images of cats and dogs. Features real-time training monitoring, GPU acceleration, and full-stack deployment.`);
      return;
    }

    alert(`${title}\nDetails coming soon.`);
  };

  return (
    <div className="page">
      <Navbar />
      <header className="header">
        <p className="subtitle">Select a base project to get started</p>
      </header>

      <section className="grid">
        {PROJECTS.map(({ id, title, subtitle, emoji, slug }) => (
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
        <a className="link" href={`${API_BASE}/api/health`} target="_blank" rel="noreferrer">
          API Health
        </a>
      </footer>
    </div>
  );
}
