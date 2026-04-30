import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { readFileSync } from 'node:fs'
import type { ServerOptions } from 'node:https'

const backendTarget = process.env.VITE_BACKEND_URL || 'http://localhost:10600'
const frontendPort = Number(process.env.VITE_FRONTEND_PORT || process.env.FRONTEND_PORT || 3443)

function localHttpsConfig(): ServerOptions | undefined {
  const certPath = process.env.VITE_DEV_SSL_CERT
  const keyPath = process.env.VITE_DEV_SSL_KEY

  if (!certPath && !keyPath) {
    return undefined
  }
  if (!certPath || !keyPath) {
    throw new Error('Both VITE_DEV_SSL_CERT and VITE_DEV_SSL_KEY are required for HTTPS dev serving')
  }

  return {
    cert: readFileSync(certPath),
    key: readFileSync(keyPath),
  }
}

const devHttps = localHttpsConfig()

export default defineConfig({
  plugins: [react()],
  server: {
    port: frontendPort,
    ...(devHttps ? { https: devHttps } : {}),
    proxy: {
      '/api': {
        target: backendTarget,
        changeOrigin: true,
        secure: false,
      },
      '/health': {
        target: backendTarget,
        changeOrigin: true,
        secure: false,
      },
      '/socket.io': {
        target: backendTarget,
        ws: true,
        secure: false,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: {
          react: ['react', 'react-dom', 'react-router-dom'],
          charts: ['chart.js', 'react-chartjs-2'],
          query: ['@tanstack/react-query'],
          vendor: ['clsx', 'lucide-react', 'socket.io-client', 'wavesurfer.js'],
        },
      },
    },
  },
})
