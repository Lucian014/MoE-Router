import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the UI runs on localhost:5173 and /api goes to the cluster through the ingress;
// in production the gateway serves both, so the code always calls relative /api/... paths.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": { target: "http://llm-router", changeOrigin: true },
    },
  },
});