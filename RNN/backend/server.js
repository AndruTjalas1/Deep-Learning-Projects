const express = require("express");
const cors = require("cors");

const app = express();
app.use(express.json());

// Allow your Vercel app + local dev
app.use(cors({
  origin: [
    "https://cst-435-react.vercel.app",
    "https://cst-435-react-b7fbctswi-tatums-projects-965c11b1.vercel.app",
    "http://localhost:5173"
  ],
  methods: ["GET","POST","OPTIONS"],
  allowedHeaders: ["Content-Type","Authorization"]
}));

// ---- ROUTES ----
// Health (your browser checks '/' and your UI checks '/api/' via rewrite)
app.get("/", (req, res) =>
  res.json({ status: "healthy", service: "rnn-api" })
);
app.get("/health", (req, res) =>
  res.json({ status: "healthy", service: "rnn-api" })
);

// Model info — support both spellings used across
