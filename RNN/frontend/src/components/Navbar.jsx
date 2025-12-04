import React from 'react';
import { FiHome } from 'react-icons/fi';
import '../styles/Navbar.css';

export default function Navbar() {
  const handleHome = () => {
    window.location.href = '/';
  };

  return (
    <nav className="navbar">
      <div className="navbar-container">
        <h1 className="navbar-title">RNN Text Generator</h1>
        <button className="navbar-home-btn" onClick={handleHome} title="Return to Home">
          <FiHome size={24} />
          <span>Home</span>
        </button>
      </div>
    </nav>
  );
}
