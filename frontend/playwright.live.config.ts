import { defineConfig } from '@playwright/test'
import { existsSync } from 'node:fs'

const FRONTEND_PORT = Number(process.env.AUTOVOICE_PLAYWRIGHT_PORT ?? 4273)
const BACKEND_PORT = Number(process.env.AUTOVOICE_LIVE_BACKEND_PORT ?? 5051)
// Canonical Jetson interpreter when present; plain python3 elsewhere (hosted
// CI runners), overridable via AUTOVOICE_PYTHON either way.
const JETSON_PYTHON = '/home/kp/anaconda3/envs/autovoice-thor/bin/python'
const BACKEND_PYTHON =
  process.env.AUTOVOICE_PYTHON ?? (existsSync(JETSON_PYTHON) ? JETSON_PYTHON : 'python3')
const HOST = '127.0.0.1'

export default defineConfig({
  testDir: './e2e',
  testMatch: /.*\.live\.spec\.ts/,
  timeout: 60_000,
  workers: 1,
  retries: 0,
  reporter: [['html', { open: 'never', outputFolder: 'playwright-report-live' }], ['list']],
  use: {
    baseURL: `http://${HOST}:${FRONTEND_PORT}`,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  webServer: [
    {
      command: `PYTHONNOUSERSITE=1 PYTHONPATH=../src ${BACKEND_PYTHON} ../tests/live_backend_server.py --host ${HOST} --port ${BACKEND_PORT}`,
      port: BACKEND_PORT,
      reuseExistingServer: false,
      timeout: 120_000,
    },
    {
      command: `VITE_BACKEND_URL=http://${HOST}:${BACKEND_PORT} npm run dev -- --host ${HOST} --port ${FRONTEND_PORT}`,
      port: FRONTEND_PORT,
      reuseExistingServer: false,
      timeout: 120_000,
    },
  ],
})
