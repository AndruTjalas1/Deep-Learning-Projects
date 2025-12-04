import React from 'react';
import '../styles/Navbar.css';

export default function Navbar() {
  return (
    <nav className="navbar">
      <div className="navbar-container">
        <div className="navbar-branding">
          <h1 className="navbar-title">CST-435</h1>
          <p className="navbar-subtitle">Machine Learning Projects</p>
        </div>
      </div>
    </nav>
  );
}
