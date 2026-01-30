import { test, expect } from '@playwright/test'

test.describe('Inference Configuration', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/convert')
  })

  test('should display inference config panel', async ({ page }) => {
    const configPanel = page.locator('[data-testid="inference-config-panel"]')
    await expect(configPanel).toBeVisible()
  })

  test('should adjust pitch shift slider', async ({ page }) => {
    const slider = page.locator('[data-testid="pitch-shift-slider"]')

    // Set to +5 semitones
    await slider.fill('5')
    await expect(slider).toHaveValue('5')

    // Set to -3 semitones
    await slider.fill('-3')
    await expect(slider).toHaveValue('-3')
  })

  test('should adjust volume sliders', async ({ page }) => {
    const vocalSlider = page.locator('[data-testid="vocal-volume-slider"]')
    const instrumentalSlider = page.locator('[data-testid="instrumental-volume-slider"]')

    await vocalSlider.fill('120')
    await instrumentalSlider.fill('80')

    await expect(vocalSlider).toHaveValue('120')
    await expect(instrumentalSlider).toHaveValue('80')
  })

  test('should select quality preset', async ({ page }) => {
    const presetSelector = page.locator('[data-testid="quality-preset-selector"]')

    await presetSelector.selectOption('studio')
    await expect(presetSelector).toHaveValue('studio')

    // Check that corresponding settings are updated
    const stepsDisplay = page.locator('[data-testid="diffusion-steps"]')
    await expect(stepsDisplay).toContainText('200')
  })

  test('should switch encoder backend', async ({ page }) => {
    const encoderSelector = page.locator('[data-testid="encoder-selector"]')

    await encoderSelector.selectOption('contentvec')
    await expect(encoderSelector).toHaveValue('contentvec')
  })

  test('should switch vocoder type', async ({ page }) => {
    const vocoderSelector = page.locator('[data-testid="vocoder-selector"]')

    await vocoderSelector.selectOption('bigvgan')
    await expect(vocoderSelector).toHaveValue('bigvgan')
  })
})
