import { test, expect } from '@playwright/test'

test.describe('Training Configuration', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/profiles')
  })

  test('should display training config panel', async ({ page }) => {
    // Navigate to a profile page
    await page.click('[data-testid="profile-card"]:first-child')

    // Check training config is visible
    const trainingPanel = page.locator('[data-testid="training-config-panel"]')
    await expect(trainingPanel).toBeVisible()
  })

  test('should update LoRA rank slider', async ({ page }) => {
    await page.click('[data-testid="profile-card"]:first-child')

    const slider = page.locator('[data-testid="lora-rank-slider"]')
    await slider.fill('16')

    await expect(slider).toHaveValue('16')
  })

  test('should toggle EWC regularization', async ({ page }) => {
    await page.click('[data-testid="profile-card"]:first-child')

    const toggle = page.locator('[data-testid="ewc-toggle"]')
    const initialState = await toggle.isChecked()

    await toggle.click()

    const newState = await toggle.isChecked()
    expect(newState).toBe(!initialState)
  })

  test('should validate learning rate input', async ({ page }) => {
    await page.click('[data-testid="profile-card"]:first-child')

    const input = page.locator('[data-testid="learning-rate-input"]')

    // Test valid scientific notation
    await input.fill('1e-4')
    await expect(input).not.toHaveClass(/error/)

    // Test invalid input
    await input.fill('invalid')
    await expect(input).toHaveClass(/error/)
  })

  test('should start training with custom config', async ({ page }) => {
    await page.click('[data-testid="profile-card"]:first-child')

    // Set custom values
    await page.locator('[data-testid="lora-rank-slider"]').fill('32')
    await page.locator('[data-testid="epochs-slider"]').fill('20')

    // Start training
    await page.click('[data-testid="start-training-btn"]')

    // Check training started
    const progressBar = page.locator('[data-testid="training-progress"]')
    await expect(progressBar).toBeVisible()
  })
})
