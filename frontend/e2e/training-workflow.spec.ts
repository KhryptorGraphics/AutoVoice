import { test, expect } from '@playwright/test'

test.describe('Full Training Workflow', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/profiles')
  })

  test('should complete full voice profile training workflow', async ({ page }) => {
    // Step 1: Create new profile
    const createBtn = page.locator('[data-testid="create-profile-btn"]')
    await createBtn.click()

    // Step 2: Enter profile details
    const nameInput = page.locator('[data-testid="profile-name-input"]')
    await nameInput.fill('Test Singer Profile')

    const descInput = page.locator('[data-testid="profile-description-input"]')
    await descInput.fill('Profile for E2E testing')

    // Step 3: Upload training samples
    const fileInput = page.locator('input[type="file"][data-testid="training-samples-input"]')
    await fileInput.setInputFiles([
      { name: 'sample1.wav', mimeType: 'audio/wav', buffer: Buffer.from('audio1') },
      { name: 'sample2.wav', mimeType: 'audio/wav', buffer: Buffer.from('audio2') },
      { name: 'sample3.wav', mimeType: 'audio/wav', buffer: Buffer.from('audio3') },
    ])

    // Verify samples uploaded
    const sampleList = page.locator('[data-testid="uploaded-sample"]')
    await expect(sampleList).toHaveCount(3)

    // Step 4: Configure training parameters
    const loraRankSlider = page.locator('[data-testid="lora-rank-slider"]')
    await loraRankSlider.fill('16')

    const epochsSlider = page.locator('[data-testid="epochs-slider"]')
    await epochsSlider.fill('10')

    // Enable EWC regularization
    const ewcToggle = page.locator('[data-testid="ewc-toggle"]')
    await ewcToggle.click()

    // Step 5: Start training
    const startBtn = page.locator('[data-testid="start-training-btn"]')
    await startBtn.click()

    // Step 6: Monitor training progress
    const progressBar = page.locator('[data-testid="training-progress"]')
    await expect(progressBar).toBeVisible()

    // Step 7: Wait for completion (with extended timeout for training)
    const completeStatus = page.locator('[data-testid="training-complete"]')
    await expect(completeStatus).toBeVisible({ timeout: 120000 })

    // Step 8: Verify profile appears in list
    await page.goto('/profiles')
    const profileCard = page.locator('[data-testid="profile-card"]').filter({ hasText: 'Test Singer Profile' })
    await expect(profileCard).toBeVisible()
  })

  test('should pause and resume training', async ({ page }) => {
    // Navigate to profile with active training
    await page.click('[data-testid="profile-card"]:first-child')

    // Start training
    const startBtn = page.locator('[data-testid="start-training-btn"]')
    await startBtn.click()

    // Wait for training to start
    const progressBar = page.locator('[data-testid="training-progress"]')
    await expect(progressBar).toBeVisible()

    // Pause training
    const pauseBtn = page.locator('[data-testid="pause-training-btn"]')
    await pauseBtn.click()

    // Verify paused state
    const pausedStatus = page.locator('[data-testid="training-paused"]')
    await expect(pausedStatus).toBeVisible()

    // Resume training
    const resumeBtn = page.locator('[data-testid="resume-training-btn"]')
    await resumeBtn.click()

    // Verify resumed
    await expect(progressBar).toBeVisible()
  })

  test('should cancel training with confirmation', async ({ page }) => {
    await page.click('[data-testid="profile-card"]:first-child')

    const startBtn = page.locator('[data-testid="start-training-btn"]')
    await startBtn.click()

    // Wait for training to start
    await expect(page.locator('[data-testid="training-progress"]')).toBeVisible()

    // Cancel training
    const cancelBtn = page.locator('[data-testid="cancel-training-btn"]')
    await cancelBtn.click()

    // Confirm cancellation
    const confirmBtn = page.locator('[data-testid="confirm-cancel-training"]')
    await confirmBtn.click()

    // Verify training stopped
    const progressBar = page.locator('[data-testid="training-progress"]')
    await expect(progressBar).not.toBeVisible()
  })

  test('should show training metrics in real-time', async ({ page }) => {
    await page.click('[data-testid="profile-card"]:first-child')

    const startBtn = page.locator('[data-testid="start-training-btn"]')
    await startBtn.click()

    // Check metrics display
    const lossChart = page.locator('[data-testid="training-loss-chart"]')
    await expect(lossChart).toBeVisible()

    const epochCounter = page.locator('[data-testid="current-epoch"]')
    await expect(epochCounter).toBeVisible()

    const gpuMemory = page.locator('[data-testid="training-gpu-memory"]')
    await expect(gpuMemory).toBeVisible()
  })

  test('should validate minimum training samples', async ({ page }) => {
    const createBtn = page.locator('[data-testid="create-profile-btn"]')
    await createBtn.click()

    // Try to start training without samples
    const startBtn = page.locator('[data-testid="start-training-btn"]')

    // Button should be disabled
    await expect(startBtn).toBeDisabled()

    // Upload only one sample (below minimum)
    const fileInput = page.locator('input[type="file"][data-testid="training-samples-input"]')
    await fileInput.setInputFiles({
      name: 'sample1.wav',
      mimeType: 'audio/wav',
      buffer: Buffer.from('audio'),
    })

    // Should show warning
    const warning = page.locator('[data-testid="minimum-samples-warning"]')
    await expect(warning).toBeVisible()
  })

  test('should compare model versions after training', async ({ page }) => {
    await page.click('[data-testid="profile-card"]:first-child')

    // Navigate to versions tab
    const versionsTab = page.locator('[data-testid="model-versions-tab"]')
    await versionsTab.click()

    // Select two versions for comparison
    const versionCheckboxes = page.locator('[data-testid="version-checkbox"]')
    await versionCheckboxes.nth(0).click()
    await versionCheckboxes.nth(1).click()

    // Start comparison
    const compareBtn = page.locator('[data-testid="compare-versions-btn"]')
    await compareBtn.click()

    // Verify comparison view
    const comparisonPanel = page.locator('[data-testid="version-comparison-panel"]')
    await expect(comparisonPanel).toBeVisible()
  })
})
