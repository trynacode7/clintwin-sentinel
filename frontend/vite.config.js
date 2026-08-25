import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Phase K note: src/hooks/usePatientData.js currently fetches the
// backend via an absolute URL (http://localhost:8000), not through this
// proxy — CORS is already open on the backend (see backend/api.py's
// CORSMiddleware), so the proxy isn't required for the app to work.
//
// This proxy config previously had a bug worth recording: it rewrote
// `/api/*` to strip the `/api` prefix before forwarding
// (`/api/roster` -> `localhost:8000/roster`), but the real backend's
// routes ARE prefixed with `/api/` (`GET /api/roster`, not `GET
// /roster`) — so a request routed through this proxy would have 404'd
// against the real endpoints. Fixed by removing the rewrite so
// `/api/*` forwards unchanged. Kept in place (rather than deleted) in
// case a future change prefers relative fetches through this proxy
// over the current absolute-URL approach.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
