import { expect, test } from '@playwright/test'

import { createWavBuffer } from './support/mockApi'

test.describe('Live backend conversion workflow', () => {
  test('submits conversion, observes completion, and sees history', async ({ page }) => {
    await page.goto('/')

    await page.locator('select').nth(1).selectOption('live-demo-singer')

    await page.locator('#artist-song-upload').setInputFiles({
      name: 'live-artist-song.wav',
      mimeType: 'audio/wav',
      buffer: createWavBuffer(),
    })
    await page.locator('#user-vocals-upload').setInputFiles({
      name: 'live-user-vocal.wav',
      mimeType: 'audio/wav',
      buffer: createWavBuffer(),
    })

    await expect(page.getByText('Ready for Conversion')).toBeVisible({ timeout: 20_000 })
    await page.getByRole('button', { name: /Convert Workflow Song/ }).click()

    await expect(page.getByText('Conversion complete', { exact: true })).toBeVisible({ timeout: 15_000 })
    await expect(page.getByRole('button', { name: 'Download Mix' })).toBeVisible()

    await page.goto('/history')
    await expect(page.getByRole('heading', { name: 'Conversion History' })).toBeVisible()
    await expect(page.getByText('Live Demo Singer')).toBeVisible()
  })
})
