import { defineConfig } from 'vitest/config';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const rootDir = dirname(fileURLToPath(import.meta.url));

// The one production entry is review.html; Vite keeps that filename in dist/ so
// the Python server can go on serving /review.html. `base: './'` makes the
// injected asset URLs relative to the page, so the same dist/ works whether the
// server serves it at /review.html or the redirect from /.
//
// OpenSeadragon rides along only for the ?viewer=osd comparison harness and is
// lazy-imported, so it never enters the production initial graph; Panzoom is the
// production viewer and is bundled directly.
export default defineConfig({
  root: rootDir,
  base: './',
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    sourcemap: false,
    reportCompressedSize: true,
    rollupOptions: {
      input: { review: resolve(rootDir, 'review.html') },
    },
  },
  // `npm run dev` serves the UI on 18930 and forwards /api to a review server
  // you start separately (e.g. `python3 frontend/harness/fixture_server.py
  // --port 18931`, or a real `run_pipeline` server). Production serves the built
  // dist/ from the Python server itself, so this proxy is dev-only.
  server: {
    port: 18930,
    strictPort: true,
    proxy: { '/api': { target: 'http://127.0.0.1:18931', changeOrigin: true } },
  },
  test: {
    environment: 'happy-dom',
    include: ['test/**/*.test.ts', 'test/**/*.test.tsx'],
    globals: false,
  },
});
