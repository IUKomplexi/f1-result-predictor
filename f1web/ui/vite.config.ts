import { defineConfig } from 'vite'
import preact from '@preact/preset-vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [preact()],
  server: {
    // Dev: forward API calls to the FastAPI backend on :8080 so the SPA can
    // always talk to the same-origin `/api` paths it will use in production.
    proxy: {
      '/api': 'http://127.0.0.1:8080',
    },
  },
})
