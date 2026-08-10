import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Dev: forward API calls to the FastAPI backend on :8080 so the SPA can
    // always talk to the same-origin `/api` paths it will use in production.
    proxy: {
      '/api': 'http://127.0.0.1:8080',
    },
  },
})
