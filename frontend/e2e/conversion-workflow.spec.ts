import { test, expect } from '@playwright/test'

test.describe('Full Conversion Workflow', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/convert')
  })

  test('should complete full voice conversion workflow', async ({ page }) => {
    // Step 1: Upload audio file
    const fileInput = page.locator('input[type="file"]')
    await fileInput.setInputFiles({
      name: 'test-song.mp3',
      mimeType: 'audio/mpeg',
      buffer: Buffer.from('mock audio content'),
    })

    // Verify file is queued
    const queueItem = page.locator('[data-testid="queue-item"]')
    await expect(queueItem).toBeVisible()

    // Step 2: Select voice profile
    const profileSelector = page.locator('[data-testid="voice-profile-selector"]')
    await profileSelector.click()
    const profileOption = page.locator('[data-testid="profile-option"]').first()
    await profileOption.click()

    // Step 3: Configure conversion settings
    const pitchSlider = page.locator('[data-testid="pitch-shift-slider"]')
    await pitchSlider.fill('2')

    const qualitySelector = page.locator('[data-testid="quality-preset-selector"]')
    await qualitySelector.selectOption('high')

    // Step 4: Start conversion
    const convertBtn = page.locator('[data-testid="start-conversion-btn"]')
    await convertBtn.click()

    // Step 5: Monitor progress
    const progressBar = page.locator('[data-testid="conversion-progress"]')
    await expect(progressBar).toBeVisible()

    // Step 6: Wait for completion (with timeout)
    const downloadBtn = page.locator('[data-testid="download-result-btn"]')
    await expect(downloadBtn).toBeVisible({ timeout: 60000 })

    // Step 7: Verify result appears in history
    await page.goto('/history')
    const historyItem = page.locator('[data-testid="history-item"]').first()
    await expect(historyItem).toBeVisible()
  })

  test('should handle conversion cancellation', async ({ page }) => {
    // Upload and start conversion
    const fileInput = page.locator('input[type="file"]')
    await fileInput.setInputFiles({
      name: 'test-song.mp3',
      mimeType: 'audio/mpeg',
      buffer: Buffer.from('mock audio'),
    })

    const convertBtn = page.locator('[data-testid="start-conversion-btn"]')
    await convertBtn.click()

    // Cancel conversion
    const cancelBtn = page.locator('[data-testid="cancel-conversion-btn"]')
    await cancelBtn.click()

    // Confirm cancellation
    const confirmCancel = page.locator('[data-testid="confirm-cancel"]')
    await confirmCancel.click()

    // Verify conversion stopped
    const progressBar = page.locator('[data-testid="conversion-progress"]')
    await expect(progressBar).not.toBeVisible()
  })

  test('should show error for invalid file format', async ({ page }) => {
    const fileInput = page.locator('input[type="file"]')
    await fileInput.setInputFiles({
      name: 'invalid.txt',
      mimeType: 'text/plain',
      buffer: Buffer.from('not audio'),
    })

    const errorMessage = page.locator('[data-testid="upload-error"]')
    await expect(errorMessage).toBeVisible()
    await expect(errorMessage).toContainText('Invalid file format')
  })

  test('should persist settings across page reload', async ({ page }) => {
    // Set custom settings
    const pitchSlider = page.locator('[data-testid="pitch-shift-slider"]')
    await pitchSlider.fill('5')

    // Reload page
    await page.reload()

    // Verify settings persisted
    await expect(pitchSlider).toHaveValue('5')
  })

  test('should show real-time GPU status during conversion', async ({ page }) => {
    const fileInput = page.locator('input[type="file"]')
    await fileInput.setInputFiles({
      name: 'test-song.mp3',
      mimeType: 'audio/mpeg',
      buffer: Buffer.from('mock audio'),
    })

    const convertBtn = page.locator('[data-testid="start-conversion-btn"]')
    await convertBtn.click()

    // GPU status should update during conversion
    const gpuStatus = page.locator('[data-testid="gpu-status-indicator"]')
    await expect(gpuStatus).toBeVisible()
  })
})
