import { expect, test } from '@playwright/test'

import { installBrowserAudioMocks } from './support/browserAudio'
import { createWavBuffer } from './support/mockApi'

test.describe('Live backend karaoke workflow', () => {
  test('uploads a song, completes preflight, and starts/stops a live session', async ({ page }) => {
    await installBrowserAudioMocks(page)
    await page.goto('/karaoke')

    await page.getByTestId('karaoke-upload-input').setInputFiles({
      name: 'karaoke-demo.wav',
      mimeType: 'audio/wav',
      buffer: createWavBuffer(24000, 2),
    })

    await expect(page.getByTestId('karaoke-separation-stage')).toBeVisible()
    await expect(page.getByTestId('karaoke-start-button')).toBeVisible({ timeout: 15_000 })

    await page.getByTestId('karaoke-speaker-device-select').selectOption('1')
    await page.getByTestId('karaoke-headphone-device-select').selectOption('0')

    await page.getByTestId('karaoke-start-button').click()
    await expect(page.getByTestId('karaoke-live-indicator')).toContainText('LIVE', { timeout: 15_000 })

    await page.getByTestId('karaoke-stop-button').click()
    await expect(page.getByTestId('karaoke-start-button')).toBeVisible()
  })
})
