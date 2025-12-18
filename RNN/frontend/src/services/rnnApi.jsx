// RNN/frontend/src/services/rnnApi.jsx
// Use environment variable for production, Vercel rewrite for deployment, localhost for dev
const base = (import.meta.env.VITE_API_BASE || "/api").replace(/\/+$/, "");

// 90 second timeout for cold starts on Azure Free Tier
async function fetchWithTimeout(url, options = {}, timeout = 90000) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);
  
  try {
    const res = await fetch(url, { ...options, signal: controller.signal });
    clearTimeout(timeoutId);
    return res;
  } catch (error) {
    clearTimeout(timeoutId);
    throw error;
  }
}

async function get(path) {
  const res = await fetchWithTimeout(`${base}${path}`, {
    method: "GET",
    headers: { "Content-Type": "application/json" },
  });
  if (!res.ok) throw new Error(`GET ${path} failed: ${res.status}`);
  return res.json();
}

async function post(path, body) {
  const res = await fetchWithTimeout(`${base}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`POST ${path} failed: ${res.status} ${text}`);
  }
  return res.json();
}

export const rnnApi = {
  health: () => get("/health"),
  modelInfo: () => get("/model-info"),

  // Frontend-friendly inputs → FastAPI schema
  generate: ({ seed, words, temperature, topK = 0, topP = 0, useBeam = false, beamWidth = 3 }) =>
    post("/generate", {
      seed_text: seed,
      num_words: Number(words),
      temperature: Number(temperature),
      top_k: Number(topK) || 0,
      top_p: Number(topP) || 0,
      use_beam_search: !!useBeam,
      beam_width: Number(beamWidth) || 3,
    }),
};
