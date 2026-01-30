import { test, expect } from '@playwright/test'

test.describe('Preset Management', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/convert')
  })

  test('should display preset manager', async ({ page }) => {
    const presetManager = page.locator('[data-testid="preset-manager"]')
    await expect(presetManager).toBeVisible()
  })

  test('should show save preset button', async ({ page }) => {
    const saveBtn = page.locator('[data-testid="save-preset-btn"]')
    await expect(saveBtn).toBeVisible()
  })

  test('should open save preset dialog', async ({ page }) => {
    await page.click('[data-testid="save-preset-btn"]')

    const dialog = page.locator('[data-testid="save-preset-dialog"]')
    await expect(dialog).toBeVisible()
  })

  test('should save new preset', async ({ page }) => {
    // Configure some settings first
    await page.locator('[data-testid="pitch-shift-slider"]').fill('3')
    await page.locator('[data-testid="quality-preset-selector"]').selectOption('high')

    // Save preset
    await page.click('[data-testid="save-preset-btn"]')
    await page.locator('[data-testid="preset-name-input"]').fill('My Test Preset')
    await page.click('[data-testid="confirm-save-preset"]')

    // Check preset appears in list
    const presetItem = page.locator('[data-testid="preset-item"]').filter({ hasText: 'My Test Preset' })
    await expect(presetItem).toBeVisible()
  })

  test('should load preset', async ({ page }) => {
    // Assuming a preset exists
    const presetItem = page.locator('[data-testid="preset-item"]').first()
    await presetItem.click()

    // Settings should be updated
    const pitchSlider = page.locator('[data-testid="pitch-shift-slider"]')
    await expect(pitchSlider).toBeVisible()
  })

  test('should delete preset', async ({ page }) => {
    const presetItem = page.locator('[data-testid="preset-item"]').first()
    const deleteBtn = presetItem.locator('[data-testid="delete-preset-btn"]')

    await deleteBtn.click()

    // Confirm deletion
    await page.click('[data-testid="confirm-delete"]')

    // Check preset is removed (or reduced count)
  })

  test('should export presets', async ({ page }) => {
    const [download] = await Promise.all([
      page.waitForEvent('download'),
      page.click('[data-testid="export-presets-btn"]'),
    ])

    expect(download.suggestedFilename()).toContain('presets')
    expect(download.suggestedFilename()).toContain('.json')
  })

  test('should import presets', async ({ page }) => {
    const importBtn = page.locator('[data-testid="import-presets-btn"]')
    const fileInput = page.locator('input[type="file"][accept=".json"]')

    await importBtn.click()
    await fileInput.setInputFiles({
      name: 'presets.json',
      mimeType: 'application/json',
      buffer: Buffer.from(JSON.stringify([
        { id: 'test-1', name: 'Imported Preset', config: { pitch_shift: 5 } }
      ])),
    })

    // Check imported preset appears
    const importedPreset = page.locator('[data-testid="preset-item"]').filter({ hasText: 'Imported Preset' })
    await expect(importedPreset).toBeVisible()
  })
})
