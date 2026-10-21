import { defineConfig } from "vitest/config";

// base: "./" makes the build work from any sub-path (GitHub Pages, a folder on your site, ...).
export default defineConfig({
  base: "./",
  build: { target: "es2022", outDir: "dist" },
  test: { environment: "node", include: ["test/**/*.test.ts"] },
});
