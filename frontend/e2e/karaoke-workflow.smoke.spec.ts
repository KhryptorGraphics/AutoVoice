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
    await expect(page.getByTestId('browser-singalong-capture')).toBeVisible()

    await page.getByTestId('karaoke-speaker-device-select').selectOption('1')
    await page.getByTestId('karaoke-headphone-device-select').selectOption('0')

    await expect.poll(() => mockedApi.getSpeakerDevice()).toBe(1)
    await expect.poll(() => mockedApi.getHeadphoneDevice()).toBe(0)

    await expect(page.getByTestId('browser-capture-input-select').locator('option')).toHaveCount(2)
    await expect(page.getByTestId('browser-capture-output-select').locator('option')).toHaveCount(2)
    await page.getByTestId('browser-capture-enable-mic').click()
    await expect(page.getByText('Browser microphone is ready.')).toBeVisible()
    await page.getByTestId('browser-capture-output-select').selectOption('browser-output-1')
    await page.getByTestId('browser-capture-start').click()
    await expect(page.getByTestId('browser-capture-recording-status')).toBeVisible()
    await page.getByTestId('browser-capture-stop').click()
    await expect(page.getByTestId('browser-capture-take-preview')).toBeVisible()
    await page.getByTestId('browser-capture-attach').click()
    await expect(page.getByText('Recorded take attached to the selected profile.')).toBeVisible()
    await expect.poll(() => mockedApi.getProfileSampleCount()).toBe(4)

    await page.getByTestId('karaoke-start-button').click()
    await expect(page.getByTestId('karaoke-live-indicator')).toContainText('LIVE')

    await page.getByTestId('karaoke-stop-button').click()
    await expect(page.getByTestId('karaoke-start-button')).toBeVisible()
  })
})
