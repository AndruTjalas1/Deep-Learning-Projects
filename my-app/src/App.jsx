import React from "react";
import "./index.css";

const PROJECTS = [
  { id: 1, title: "Project 1", subtitle: "Starter Template", emoji: "🚀", slug: "project-1" },
  { id: 2, title: "Project 2", subtitle: "Data Pipeline", emoji: "📊", slug: "project-2" },
  { id: 3, title: "Project 3", subtitle: "API Service", emoji: "🔌", slug: "project-3" },
  { id: 4, title: "Project 4", subtitle: "ML Model", emoji: "🤖", slug: "project-4" },
  { id: 5, title: "Project 5", subtitle: "UI Components", emoji: "🎨", slug: "project-5" },
  { id: 6, title: "Project 6", subtitle: "Auth & Users", emoji: "🛂", slug: "project-6" },
  { id: 7, title: "Project 7", subtitle: "Testing Suite", emoji: "🧪", slug: "project-7" },
  { id: 8, title: "Project 8", subtitle: "Deployment", emoji: "☁️", slug: "project-8" },
];

export default function App() {
  const handleLaunch = (slug) => {
    window.location.href = `/project/${slug}`;
  };

  const handleDetails = (title) => {
    alert(`${title}\nDetails coming soon…`);
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
              <button className="btn primary" onClick={() => handleLaunch(slug)}>Launch</button>
              <button className="btn ghost" onClick={() => handleDetails(title)}>Details</button>
            </div>
          </article>
        ))}
      </section>

      <footer className="footer">
        <a className="link" href="/api/health" target="_blank" rel="noreferrer">API Health</a>
      </footer>
    </div>
  );
}
