import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  root: 'frontend',
  // Relative asset paths: the static bundle works from any host or sub-path (e.g. GitHub Pages)
  base: './',
  server: {
    port: 3000,
    open: false,
  },
  // The engine runs in a module Web Worker that imports the WebAssembly bindings
  worker: {
    format: 'es',
  },
  build: {
    outDir: '../dist',
    emptyOutDir: true,
  },
});
