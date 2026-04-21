import { expect, test } from '@playwright/test'

import { createWavBuffer } from './support/mockApi'

test.describe('Live backend conversion workflow', () => {
  test('submits conversion, observes completion, and sees history', async ({ page }) => {
    await page.goto('/')

    await page.getByTestId('conversion-audio-input').setInputFiles({
      name: 'live-conversion.wav',
      mimeType: 'audio/wav',
      buffer: createWavBuffer(),
    })

    await page.getByTestId('voice-profile-selector').selectOption('live-demo-singer')
    await page.getByTestId('start-conversion-button').click()

    await expect(page.getByText('Conversion Complete!')).toBeVisible({ timeout: 15_000 })
    await expect(page.getByRole('button', { name: 'Download Mix' })).toBeVisible()

    await page.goto('/history')
    await expect(page.getByRole('heading', { name: 'Conversion History' })).toBeVisible()
    await expect(page.getByText('Live Demo Singer')).toBeVisible()
  })
})
