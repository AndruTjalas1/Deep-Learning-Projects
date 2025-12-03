import { defineConfig } from "vite";
// add plugins like react if this is a React app:
// import react from "@vitejs/plugin-react";
export default defineConfig({
  // plugins: [react()],
  base: "/rnn/",
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:5001',
        changeOrigin: true,
      }
    }
  }
});
