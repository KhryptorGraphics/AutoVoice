import { expect, test } from '@playwright/test'

import { mockCommonApi } from './support/mockApi'

test.describe('Karaoke workflow smoke', () => {
  test('covers upload, separation, device controls, and live start/stop', async ({ page }) => {
    const mockedApi = await mockCommonApi(page)

    await page.goto('/karaoke')

    await page.getByTestId('karaoke-upload-input').setInputFiles({
      name: 'pillowtalk.wav',
      mimeType: 'audio/wav',
      buffer: Buffer.alloc(2048),
    })

    await expect(page.getByTestId('karaoke-separation-stage')).toBeVisible()
    await expect(page.getByTestId('karaoke-voice-model-select')).toBeVisible()

    await page.getByTestId('karaoke-speaker-device-select').selectOption('1')
    await page.getByTestId('karaoke-headphone-device-select').selectOption('0')

    await expect.poll(() => mockedApi.getSpeakerDevice()).toBe(1)
    await expect.poll(() => mockedApi.getHeadphoneDevice()).toBe(0)

    await page.getByTestId('karaoke-start-button').click()
    await expect(page.getByTestId('karaoke-live-indicator')).toContainText('LIVE')

    await page.getByTestId('karaoke-stop-button').click()
    await expect(page.getByTestId('karaoke-start-button')).toBeVisible()
  })
})
