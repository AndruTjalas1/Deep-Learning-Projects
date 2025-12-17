import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  base: '/dnp/',
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: false,
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
  },
  define: {
    'process.env.VITE_API_URL3': JSON.stringify(process.env.VITE_API_URL3 || 'http://localhost:5000'),
  }
})
