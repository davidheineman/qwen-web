import { resolve } from "node:path";
import { defineConfig } from "vite";

export default defineConfig(({ command }) => ({
  // Relative asset paths so the build works under any sub-path (e.g. GitHub Pages /qwen-web/).
  base: "./",
  server: { port: 5173 },
  // public/ only holds an optional local copy of the model weights for development; don't ship it.
  build: {
    target: "es2022",
    copyPublicDir: false,
    rollupOptions: {
      input: {
        main: resolve(__dirname, "index.html"),
        interp: resolve(__dirname, "interp.html"),
        rl: resolve(__dirname, "rl.html"),
        data: resolve(__dirname, "data.html"),
      },
    },
  },
  worker: { format: "es" },
  // Dev server only: hand HF_TOKEN from your shell to the Ladder page so the private ladder repos just work.
  // Production builds always get an empty string, so the token never ends up in dist/.
  define: {
    __DEV_HF_TOKEN__: JSON.stringify(command === "serve" ? (process.env.HF_TOKEN ?? "") : ""),
  },
}));
