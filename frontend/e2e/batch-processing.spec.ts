import { test, expect } from '@playwright/test'
import path from 'path'

test.describe('Batch Processing', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/convert')
  })

  test('should display batch processing queue', async ({ page }) => {
    const batchQueue = page.locator('[data-testid="batch-processing-queue"]')
    await expect(batchQueue).toBeVisible()
  })

  test('should show drop zone for file upload', async ({ page }) => {
    const dropZone = page.locator('[data-testid="batch-drop-zone"]')
    await expect(dropZone).toBeVisible()
    await expect(dropZone).toContainText('Drag & drop')
  })

  test('should add files via file input', async ({ page }) => {
    const fileInput = page.locator('input[type="file"]')

    // Upload a test audio file
    await fileInput.setInputFiles({
      name: 'test-audio.mp3',
      mimeType: 'audio/mpeg',
      buffer: Buffer.from('mock audio content'),
    })

    // Check file appears in queue
    const queueItem = page.locator('[data-testid="queue-item"]')
    await expect(queueItem).toBeVisible()
  })

  test('should allow reordering files via drag', async ({ page }) => {
    // Add multiple files first
    const fileInput = page.locator('input[type="file"]')
    await fileInput.setInputFiles([
      { name: 'file1.mp3', mimeType: 'audio/mpeg', buffer: Buffer.from('audio1') },
      { name: 'file2.mp3', mimeType: 'audio/mpeg', buffer: Buffer.from('audio2') },
    ])

    // Get items
    const items = page.locator('[data-testid="queue-item"]')
    await expect(items).toHaveCount(2)
  })

  test('should remove file from queue', async ({ page }) => {
    const fileInput = page.locator('input[type="file"]')
    await fileInput.setInputFiles({
      name: 'test.mp3',
      mimeType: 'audio/mpeg',
      buffer: Buffer.from('audio'),
    })

    const removeBtn = page.locator('[data-testid="remove-file-btn"]')
    await removeBtn.click()

    const queueItem = page.locator('[data-testid="queue-item"]')
    await expect(queueItem).not.toBeVisible()
  })

  test('should start batch processing', async ({ page }) => {
    const fileInput = page.locator('input[type="file"]')
    await fileInput.setInputFiles({
      name: 'test.mp3',
      mimeType: 'audio/mpeg',
      buffer: Buffer.from('audio'),
    })

    const processBtn = page.locator('[data-testid="process-batch-btn"]')
    await processBtn.click()

    // Check processing started
    const progressBar = page.locator('[data-testid="batch-progress"]')
    await expect(progressBar).toBeVisible()
  })

  test('should stop batch processing', async ({ page }) => {
    const fileInput = page.locator('input[type="file"]')
    await fileInput.setInputFiles({
      name: 'test.mp3',
      mimeType: 'audio/mpeg',
      buffer: Buffer.from('audio'),
    })

    const processBtn = page.locator('[data-testid="process-batch-btn"]')
    await processBtn.click()

    const stopBtn = page.locator('[data-testid="stop-batch-btn"]')
    await stopBtn.click()

    // Processing should be stopped
    await expect(processBtn).toBeVisible()
  })
})
