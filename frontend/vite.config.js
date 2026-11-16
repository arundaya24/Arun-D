import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Semua request /api diteruskan ke FastAPI, jadi tidak ada masalah CORS saat dev.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
});
