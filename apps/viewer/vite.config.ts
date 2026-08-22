import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Tauri serves the dev server over a fixed port and expects a relative base
// so the built bundle loads from the app's own protocol rather than a host.
export default defineConfig({
  plugins: [react()],
  base: "./",
  clearScreen: false,
  server: {
    port: 5183,
    strictPort: true,
  },
  build: {
    outDir: "dist",
    // Tauri targets a known WebView, so there is no reason to ship legacy
    // syntax or the extra bytes of a transpiled bundle.
    target: "es2022",
    sourcemap: true,
  },
});
