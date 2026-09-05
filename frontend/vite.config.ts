import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The backend base URL is read at runtime from VITE_API_BASE_URL (see src/api.ts),
// defaulting to http://localhost:8000.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173 },
})
