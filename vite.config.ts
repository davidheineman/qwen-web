import { defineConfig } from "vite";

export default defineConfig({
  // Relative asset paths so the build works under any sub-path (e.g. GitHub Pages /qwen-web/).
  base: "./",
  server: { port: 5173 },
  // public/ only holds an optional local copy of the model weights for development; don't ship it.
  build: { target: "es2022", copyPublicDir: false },
});
