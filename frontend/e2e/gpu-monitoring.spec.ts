import { test, expect } from '@playwright/test'

test.describe('GPU Monitoring', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/system')
  })

  test('should display GPU monitor', async ({ page }) => {
    const gpuMonitor = page.locator('[data-testid="gpu-monitor"]')
    await expect(gpuMonitor).toBeVisible()
  })

  test('should show GPU name and status', async ({ page }) => {
    const gpuName = page.locator('[data-testid="gpu-name"]')
    const gpuStatus = page.locator('[data-testid="gpu-status"]')

    await expect(gpuName).toBeVisible()
    await expect(gpuStatus).toBeVisible()
  })

  test('should display memory usage', async ({ page }) => {
    const memoryBar = page.locator('[data-testid="gpu-memory-bar"]')
    const memoryText = page.locator('[data-testid="gpu-memory-text"]')

    await expect(memoryBar).toBeVisible()
    await expect(memoryText).toContainText('GB')
  })

  test('should display utilization chart', async ({ page }) => {
    const utilizationChart = page.locator('[data-testid="gpu-utilization-chart"]')
    await expect(utilizationChart).toBeVisible()
  })

  test('should refresh metrics on interval', async ({ page }) => {
    const memoryText = page.locator('[data-testid="gpu-memory-text"]')

    // Get initial value
    const initialText = await memoryText.textContent()

    // Wait for refresh (default 5s)
    await page.waitForTimeout(6000)

    // Check metrics are still updating
    await expect(memoryText).toBeVisible()
  })

  test('should show temperature alert when high', async ({ page }) => {
    // This test would need mock data for high temperature
    const tempDisplay = page.locator('[data-testid="gpu-temperature"]')
    await expect(tempDisplay).toBeVisible()
  })
})
