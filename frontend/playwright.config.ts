import { defineConfig } from '@playwright/test'

const PLAYWRIGHT_PORT = Number(process.env.AUTOVOICE_PLAYWRIGHT_PORT ?? 4273)
const PLAYWRIGHT_HOST = '127.0.0.1'
const PLAYWRIGHT_BASE_URL = `http://${PLAYWRIGHT_HOST}:${PLAYWRIGHT_PORT}`

export default defineConfig({
  testDir: './e2e',
  testMatch: /.*\.smoke\.spec\.ts/,
  timeout: 30_000,
  retries: 0,
  reporter: [['html', { open: 'never', outputFolder: 'playwright-report' }], ['list']],
  use: {
    baseURL: PLAYWRIGHT_BASE_URL,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  webServer: {
    command: `npm run dev -- --host ${PLAYWRIGHT_HOST} --port ${PLAYWRIGHT_PORT}`,
    port: PLAYWRIGHT_PORT,
    // Always boot AutoVoice's own Vite server so we never latch onto an unrelated app.
    reuseExistingServer: false,
    timeout: 120_000,
  },
})
